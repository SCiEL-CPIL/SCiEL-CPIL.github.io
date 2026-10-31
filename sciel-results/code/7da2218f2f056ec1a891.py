"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: First Fit.
    We emulate the First Fit algorithm by assigning higher scores to bins
    with lower indices among those that can accommodate the item.
    This preserves bins with higher indices (typically those created later
    or with more remaining capacity in a standard list representation)
    for potentially larger future items, which helps reduce fragmentation
    compared to greedy space-minimization heuristics.
    """
    # bins is assumed to be the remaining capacity of each bin.
    # Feasible if bins[i] >= item.
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if np.any(feasible):
        # Get the indices of feasible bins
        feasible_indices = np.where(feasible)[0]
        
        # Assign scores such that the first feasible bin (lowest index) has the highest score.
        # Using negative index ensures index 0 > index 1 > ...
        # This strictly implements First Fit selection.
        scores[feasible] = -feasible_indices
        
    return scores
# EVOLVE-BLOCK-END
