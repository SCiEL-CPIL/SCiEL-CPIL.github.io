"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        # Hybrid heuristic: primarily Best Fit, with a small preference for larger bins (Worst Fit)
        # to reduce fragmentation.
        # Score = -waste + 0.1 * capacity
        scores[valid] = - remaining[valid] + 0.1 * bins[valid]
    return scores
# EVOLVE-BLOCK-END
