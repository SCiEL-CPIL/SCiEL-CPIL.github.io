"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a Worst-Fit style heuristic.
    The item goes to the highest-scoring bin.
    
    For worst-fit, we prefer bins with the most remaining capacity after placing the item.
    This preserves bins with less remaining space for smaller future items, potentially
    reducing fragmentation compared to Best-Fit.
    
    Score is the remaining capacity (bins - item) for bins that can fit the item.
    Bins that cannot fit the item get -inf.
    """
    remaining = bins - item
    # Prefer bins with the largest remaining capacity (Worst-Fit)
    # If remaining < 0, the item doesn't fit, so score is -inf.
    score = np.where(remaining >= 0, remaining, -np.inf)
    return score
# EVOLVE-BLOCK-END
