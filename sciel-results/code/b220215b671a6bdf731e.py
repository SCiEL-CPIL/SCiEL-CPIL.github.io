"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    We choose the bin with the maximum remaining space after placing the item.
    This preserves large empty spaces in other bins, which is beneficial for
    accommodating large future items and reducing fragmentation compared to
    Best Fit which can create many small gaps.
    
    bins: array of remaining capacities.
    item: size of the current item.
    
    Feasible bins: bins >= item.
    Score: remaining capacity after placement (bins - item).
    Higher score = more remaining space = preferred.
    """
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Worst Fit: Maximize the remaining space.
        # This is equivalent to picking the bin with the largest initial free space.
        scores[feasible] = remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
