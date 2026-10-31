"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Worst-Fit.
    Instead of Best-Fit (minimizing remaining space), we use Worst-Fit
    (maximizing remaining space). This tends to keep bins 'open' with more
    free space, which can be beneficial for accommodating larger items
    that might arrive later, potentially reducing the total number of
    bins used compared to Best-Fit in certain distributions like Weibull.
    """
    # Compute remaining capacity after placing the item
    remaining_after = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining_after >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, use Worst-Fit: prefer largest remaining_after
    # This maximizes the remaining capacity, keeping bins less fragmented
    # in a way that preserves large contiguous free spaces.
    scores[valid_mask] = remaining_after[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
