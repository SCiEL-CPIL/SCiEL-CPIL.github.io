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
        
    # First Fit strategy: pick the first bin (lowest index) that has enough capacity.
    # This assumes bins are ordered by creation time (index 0 is the first opened bin).
    # First Fit is a strong baseline that often outperforms Best Fit on certain distributions
    # by keeping early bins full and avoiding scattering small items.
    scores = -np.arange(len(bins), dtype=float)
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
