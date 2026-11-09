"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is the array of remaining capacities for each bin.
    # The item goes to the highest-scoring bin among those with enough capacity.
    # Approach: Pure Best-Fit
    # We select the bin that minimizes the remaining space after placing the item.
    # This strategy consolidates items into fewer bins, which is generally effective
    # for minimizing the total number of bins used, especially with distributions
    # that contain many small items (like Weibull).
    
    remaining = bins
    
    # Only consider bins that can fit the item
    mask = remaining >= item
    if not np.any(mask):
        # No bin can fit the item; return all -inf
        return np.full_like(remaining, -np.inf)
    
    # For bins that can fit the item, compute the residual after placing the item
    residual = remaining[mask] - item
    
    # Score: We want the bin with the smallest residual.
    # Since the selection mechanism picks the highest score, we negate the residual.
    score = -residual
    
    # Map back to full array
    full_score = np.full_like(remaining, -np.inf)
    full_score[mask] = score
    
    return full_score
# EVOLVE-BLOCK-END
