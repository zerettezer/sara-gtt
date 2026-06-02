# SARA — Smart Auto Response Algorithm

SARA is a research project exploring a compact, from-scratch conversational language model built on TensorFlow/Keras. The goal was to push the limits of a small model trained on a single consumer GPU, experimenting with custom components instead of relying on existing model families.

---

## Highlights

### Fully Custom Architecture
Every component — attention, feed-forward, normalization, positional encoding, loss function — was designed and implemented from scratch. No component was borrowed from an existing model family.

### Synbedding — Semantic Prior Embedding
A secondary frozen embedding generated before training using a spectral encoding method (DCT-based) over the character-level signal of each token. It provides the model with a language-agnostic semantic prior from the first training step, without any pre-training on external corpora.

- Supports 5+ writing systems natively: Latin, Cyrillic, CJK, Arabic, Greek
- Spectral whitening ensures uniform information density across the embedding space
- Fully deterministic and reproducible — no external models required

### SOFA — Adaptive Learning Rate Optimizer
A custom meta-learning rate scheduler that treats the learning rate itself as an optimization problem.

- Runs a lightweight search algorithm alongside training
- Automatically transitions between global exploration, local search, and fine-tuning phases
- Adapts based on real loss signals, gradient norms, and accuracy — not a fixed schedule
- Parameters are configurable at runtime without stopping training
- Saves a full training history chart and CSV log at the end of each run

### ShutterGLU — Custom Mixture-of-Experts FFN
A custom feed-forward block with soft expert routing: all experts contribute to every token with no hard top-k dropping. Routing temperature is a learned parameter. The entire block is XLA/JIT compiled.

### Token-Weighted Loss with Diversity Penalty
The loss function extends standard cross-entropy with:
- Per-token importance weights to prevent rare tokens from being ignored
- A batch diversity penalty that discourages training steps dominated by a narrow token distribution

### ALBERT-style Factorized Embeddings
Embedding matrices are factorized into a small inner dimension projected up to the model dimension, reducing the parameter count for large vocabularies without loss of expressiveness.

### Profile System
Training configurations can be saved and restored as profiles, making it straightforward to reproduce experiments or queue multiple training runs.

---

## Architecture

The model is an encoder-decoder transformer with a parallel local-context stream injected into each decoder block.

**Attention.** Keys and values are compressed into a shared latent via a learned bottleneck (similar to MLA), normalized, and then decompressed per-head. Queries and keys are projected with a SwiGLU gate and normalized with QK-Norm before the dot product. Positional information is injected via Rotary Position Embeddings (RoPE). A single cross-attention pass over the encoder output is computed once per sequence and cached — decoder blocks receive the result directly, eliminating redundant computation.

**Feed-forward.** Two variants are supported. ShutterGLU is a soft Mixture-of-Experts block with 4 experts, SwiGLU activations, a two-layer nonlinear router, and a learnable routing temperature that the model can sharpen or broaden during training. MultiBranchDFF is a multi-branch GLU FFN used as a leaner alternative. Both expand by a factor of 4× relative to the embedding dimension.

**Parallel stream.** Each decoder block contains a secondary branch that mixes the current decoder state with the fixed input embedding via a low-rank bilinear projection, then applies a dynamic local convolution with a query-conditioned output gate. The result is added to the main residual stream with a learned scale.

**Normalization and stability.** Pre-norm (RMSNorm) is applied before every sub-layer. Depth-aware residual scaling (MAGNETO style) initializes each sub-layer output scale at `1/sqrt(2N)`, keeping residual-stream variance approximately constant with depth. QK-Norm bounds attention logit magnitude and prevents entropy collapse.

**Embeddings.** Token embeddings are factorized (ALBERT style): a small inner dimension is projected up to the model dimension, and the same projection is tied to the output logit layer. The synbedding uses a separate factorized frozen matrix initialized from precomputed DCT encodings of each token.

**Training.** Adam (β₁=0.9, β₂=0.98, ε=1e-9) with gradient clipping. Learning rate managed by SOFA. All layers compiled with XLA JIT (`jit_compile=True`). float32 throughout.

| Component | Details |
|---|---|
| Framework | TensorFlow / Keras |
| Compilation | XLA JIT across all layers |
| Optimizer | Adam + SOFA meta-scheduler |
| Attention | Compressed KV latent, SwiGLU Q gate, RoPE, QK-Norm |
| FFN | ShutterGLU (soft MoE, 4 experts) or MultiBranchDFF |
| Normalization | RMSNorm, pre-norm, depth-aware residual scaling |
| Embeddings | ALBERT factorized, weight-tied output, frozen synbedding |
| Hardware | NVIDIA Quadro P6000, 23 GB VRAM, float32, single GPU |

---

## Results

51+ training snapshots were produced across the research period. All configs are preserved in `models/*_config.json`. Selected runs below.

**v3.0.9 — minimal configuration (1×1, d=64)**

| Parameter | Value |
|---|---|
| Encoder / decoder layers | 1 / 1 |
| Embedding dim | 64 |
| Attention heads | 1 |
| Parameters | 528,940 |
| Vocab size | 3,426 |
| Max sequence length | 48 |
| Dataset tokens | 2,449,536 |
| Training time | 0.12 h |
| Final loss / val loss | 1.029 / 0.916 |
| Final accuracy / val accuracy | 74.9% / 78.3% |
| Compute (PFLOPs) | 3.48 |
| Cost | €0.01 |

**v3.0.0 — 3×3, d=192**

| Parameter | Value |
|---|---|
| Encoder / decoder layers | 3 / 3 |
| Embedding dim | 192 |
| Attention heads | 8 |
| Parameters | 5,782,173 |
| Vocab size | 9,604 |
| Max sequence length | 144 |
| Dataset tokens | 124,762,320 |
| Training time | 34.5 h |
| Chinchilla ratio | 21.6× |
| Compute (PFLOPs) | 1,181.5 |
| Cost | €3.25 |

**v3.0.51 — 3×3, d=256, ALBERT factorization (latest)**

| Parameter | Value |
|---|---|
| Encoder / decoder layers | 3 / 3 |
| Embedding dim | 256 (ALBERT inner: 64, synbedding inner: 32) |
| Attention heads | 8 |
| Parameters | 4,274,575 |
| Vocab size | 8,456 |
| Max sequence length | 128 |
| Dataset tokens | 61,577,344 |
| Batch size | 256 |
| Steps per epoch | 1,879 |
| Training time | 17.8 h |
| Final loss / val loss | 0.469 / 0.426 |
| Final accuracy / val accuracy | 90.3% / 92.0% |
| Chinchilla ratio | 14.4× |
| Compute (PFLOPs) | 558.9 |
| Cost | €1.63 |

The v3.0.51 run converges from an initial loss of 3.83 to 0.47 over 100 epochs with SOFA keeping the learning rate in the range 3.9×10⁻⁴ – 5.8×10⁻⁴. Validation accuracy consistently exceeds training accuracy throughout, indicating no overfitting at this dataset scale.

---

## Notes on Model Configs

Not all config files in `models/` are relevant or contain sufficient information for analysis. Several categories of incomplete records exist:

### Aborted / Incomplete Runs
Some configs were written at the start of a training run that was interrupted before saving weights. These can be identified by:
- `"training_time_hours": 0.0`
- `"weights_hash": ""`
- `"p_flop_total": 0` and `"euro": 0`

Known examples: `v3.0.2`, `v3.0.13`, `v3.0.38`. These runs have no corresponding `.h5` checkpoint and cannot be used for evaluation or comparison.

### Pre-ALBERT Architecture Configs (v3.0.0 – v3.0.11)
Configs from the early phase of development predate the introduction of ALBERT-style factorized embeddings. They are missing several fields that were added later:
- `albert_emb_dim` / `albert_syn_emb_dim` — not present
- `embedding_proj` / `synbedding_proj` layers — absent from `layer_names`
- `embedding_norm` — absent or uses a different normalization type (`RMSNorm` vs `LayerNormalization`)

Comparing parameter counts or architecture details between these configs and later ones requires accounting for these structural differences.

### Known Data Quality Issues
- `vocab_size` varies significantly across runs (3,426 – 9,604), reflecting different tokenizer snapshots used during the research period. Cross-run loss comparisons are only meaningful between runs that share the same tokenizer.
