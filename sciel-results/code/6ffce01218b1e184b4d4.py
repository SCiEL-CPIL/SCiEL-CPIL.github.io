"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Refined Best Fit with slight preference for bins with moderate remaining space
    # score = -(bins - item) + alpha * bins
    # = -bins + item + alpha * bins
    # = item + (alpha - 1) * bins
    # For alpha = 0.05: score = item - 0.95 * bins
    # This is slightly less aggressive than pure Best Fit, which may reduce fragmentation
    alpha = 0.05
    scores = item + (alpha - 1.0) * bins
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
