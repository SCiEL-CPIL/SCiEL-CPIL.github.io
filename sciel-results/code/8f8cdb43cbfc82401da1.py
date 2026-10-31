"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Pure Best Fit with deadline guard.
    Best Fit minimizes the remaining capacity after placement,
    which maximizes bin fill and minimizes the number of bins used.
    """
    # Deadline check: fall back to a trivially safe Best Fit
    if time.time() > DEADLINE - SAFETY_MARGIN:
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=np.float64)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=np.float64)

    # Best Fit: score = -(remaining capacity after placement)
    # Higher score = smaller remaining = tighter fit = better
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    scores[feasible] = -(bins[feasible] - item)

    return scores
# EVOLVE-BLOCK-END
