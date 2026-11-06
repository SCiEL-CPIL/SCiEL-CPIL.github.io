"""Online bin packing – priority scoring for FunSearch/EoH contract."""
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin that has enough remaining capacity for `item`.
    The item is placed in the bin with the highest score.

    Strategy: Worst-Fit (Maximize Remaining Capacity).
    We prefer bins with the most available space after placing the item.
    This keeps bins as empty as possible, reserving space for larger
    future items and reducing fragmentation.
    
    Score = remaining capacity = bins - item.
    """
    remaining = bins - item
    mask = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    scores[mask] = remaining[mask]
    
    return scores
