"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a quadratic Best-Fit heuristic.
    
    This approach strongly penalizes bins with large remaining capacity after
    placing the item, effectively minimizing the 'waste' in the chosen bin.
    By using a quadratic penalty, the preference for the tightest fit is
    sharpened compared to a linear penalty, which can help close bins more
    efficiently and reduce the total number of bins used.
    """
    rem = bins - item
    
    # Mask for valid bins (enough capacity)
    valid = rem >= 0
    
    # For invalid bins, assign very low score
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    
    # Quadratic Best Fit:
    # Score is negative square of the remaining space.
    # This maximizes the score for the smallest remainder.
    # The quadratic term ensures that larger remainders are penalized
    # significantly more than smaller ones, encouraging tight packing.
    scores[valid] = -valid_rem**2
    
    return scores
# EVOLVE-BLOCK-END
