"""Cap set greedy construction with structured priority."""
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Priority: vectors with more non-zero entries are safer (less likely to form
    # 3-term APs with other vectors). Use count of 1s and 2s, plus a small
    # deterministic perturbation to break ties and explore different structures.
    nonzero = 0
    for v in vector:
        if v != 0:
            nonzero += 1
    # Base priority: more non-zeros = higher priority
    base = nonzero * 1.0
    # Add a small hash-based perturbation to break symmetry and explore
    # different cap set structures
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    perturbation = (h % 10000) / 10000.0 * 0.1
    return base + perturbation
# EVOLVE-BLOCK-END
