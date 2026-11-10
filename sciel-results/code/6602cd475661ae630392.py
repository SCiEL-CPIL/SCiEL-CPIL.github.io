"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Returns an n x n matrix where entry [i, j] is the attractiveness of placing
    item j right after item i in the same bin.
    
    Strategy: Strongly prefer items that fill the remaining space in the bin.
    If item i is placed, remaining space is (capacity - demands[i]).
    We want to pick j such that demands[j] is close to (capacity - demands[i]).
    
    We use a smooth inverse-gap heuristic:
    - If demands[i] + demands[j] > capacity: not feasible, set to 1e-9
    - Else: attractiveness = 1 / (1 + gap^2) where gap = capacity - demands[i] - demands[j]
      This heavily rewards exact fits (gap=0) and penalizes large gaps.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute pair sums
    pair_sum = d[:, None] + d[None, :]
    
    # Remaining space if i is placed, then we want to add j
    # gap = capacity - (d[i] + d[j])
    gap = capacity - pair_sum
    
    # Only feasible if gap >= 0
    feasible = gap >= 0
    
    # Inverse gap heuristic: 1 / (1 + gap^2) for feasible pairs
    # This gives high values for small gaps (good fill) and low values for large gaps
    # But we also want to ensure that the heuristic is positive and well-scaled
    
    # Use a sharper penalty for large gaps
    # attractiveness = exp(-alpha * gap) might be smoother
    # Let's try: 1 / (1 + (gap / scale)^2) with a small scale to sharpen
    
    scale = 10.0  # Sharper penalty for gaps > 10
    attr = 1.0 / (1.0 + (gap / scale) ** 2)
    
    # For infeasible pairs, set to 1e-9
    result = np.where(feasible, attr, 1e-9)
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
