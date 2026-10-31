"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit: prefer bins that leave the least remaining capacity.
    # We score by negative remaining space, so the highest score is the bin
    # with the smallest remaining capacity that can still fit the item.
    remaining = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    feasible = remaining >= 0
    # Among feasible bins, pick the one with the least remaining capacity
    # (i.e., the best fit). Score = -remaining, so higher score = better fit.
    scores[feasible] = -remaining[feasible]
    return scores
# EVOLVE-BLOCK-END
