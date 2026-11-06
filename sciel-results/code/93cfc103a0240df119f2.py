"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Best Fit
    We prefer bins with the smallest remaining capacity after placing the item.
    This packs items tightly, minimizing wasted space in partially filled bins.
    This is a standard heuristic that often outperforms Worst Fit on skewed distributions
    like Weibull by filling gaps efficiently.
    """
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Create a mask for bins that can fit the item
    mask = remaining >= 0
    
    # Apply the score: we want the largest remaining capacity to have the highest score.
    # This is the Worst Fit heuristic, which keeps bins "open" for larger future items.
    # For invalid bins, score is -inf.
    scores = np.full_like(remaining, -np.inf, dtype=float)
    scores[mask] = remaining[mask]
    
    return scores
# EVOLVE-BLOCK-END
