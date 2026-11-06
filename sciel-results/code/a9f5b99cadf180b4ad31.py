"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit
    We prefer bins with the largest remaining capacity after placing the item.
    This helps keep smaller bins "full" and larger bins "open" for potentially larger future items,
    reducing fragmentation compared to Best Fit.
    """
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Create a mask for bins that can fit the item
    # If a bin cannot fit the item, its score should be very low (negative infinity)
    # so it is never chosen.
    mask = remaining >= 0
    
    # Apply the score: we want the highest remaining capacity
    # So we return the remaining capacity for valid bins, and -inf for invalid ones.
    scores = np.full_like(remaining, -np.inf, dtype=float)
    scores[mask] = remaining[mask]
    
    return scores
# EVOLVE-BLOCK-END
