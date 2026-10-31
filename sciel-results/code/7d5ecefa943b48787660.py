"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Pure Best-Fit.
    - Always place the item in the bin with the smallest remaining capacity
      that can still fit the item.
    - This minimizes fragmentation and is generally optimal for minimizing
      the number of bins used, especially for distributions with many small items.
    """
    # Calculate remaining capacity if we place the item in each bin
    remaining = bins - item
    
    # Valid bins are those with remaining >= 0
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf)
    
    # If no valid bin, return all -inf
    if not np.any(valid_mask):
        return scores
    
    # Best-Fit: We want to minimize 'remaining'.
    # So we assign a score that is higher for smaller 'remaining'.
    # score = -remaining
    # Since remaining >= 0, scores will be <= 0.
    # The bin with the smallest remaining will have the largest (least negative) score.
    scores[valid_mask] = -remaining[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
