"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a variance-maximization model.
    
    The approach models bin selection to maximize the variance of the remaining 
    capacity distribution across all open bins. Maximizing variance encourages 
    polarization of bin states (either nearly full or nearly empty), which reduces 
    the number of partially filled bins and minimizes the total number of bins used.
    
    Mathematically, maximizing the variance of the remaining capacities after placing 
    the item is equivalent to selecting the bin with the smallest remaining capacity 
    that can still accommodate the item (Best Fit).
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    # If no valid bins, return all -inf
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, the score is derived from maximizing the new variance.
    # As derived, maximizing new variance is equivalent to minimizing the 
    # remaining capacity of the chosen bin (Best Fit).
    # Score = -remaining_capacity (higher score for smaller remaining capacity)
    scores[valid_mask] = -remaining[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
