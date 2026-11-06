"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a combination of:
    1. Best-fit preference (minimize leftover space)
    2. Penalize bins that are too empty (waste)
    3. Prefer bins where the item fills a significant portion
    
    The score balances tight fit with avoiding creating many nearly-empty bins.
    """
    rem = bins - item
    
    # Mask for valid bins (enough capacity)
    valid = rem >= 0
    
    # For invalid bins, assign very low score
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    valid_bins = bins[valid]
    
    # Try Worst Fit: maximize remaining capacity
    # This keeps bins "open" for larger future items, potentially reducing fragmentation
    # for distributions with significant variance like Weibull.
    scores[valid] = valid_rem
    
    return scores
# EVOLVE-BLOCK-END
