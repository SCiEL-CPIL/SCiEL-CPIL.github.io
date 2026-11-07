"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Use a tightness-weighted prize-to-cost ratio.
    - Total prize per item: sum of prizes across constraints
    - Tightness-weighted cost: weight each constraint's consumption by how 
      tight that constraint is (total demand / capacity)
    - Heuristic: total_prize / (tightness_weighted_cost + epsilon)
    
    This penalizes items that consume resources in already-tight constraints,
    guiding ants toward items that are both valuable and resource-efficient
    given the overall problem structure.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Total prize per item (sum across 5 constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Tightness of each constraint: total demand / capacity (capacity = 25)
    total_demand = weights.sum(axis=1)  # shape (5,)
    tightness = total_demand / 25.0      # shape (5,)
    
    # Tightness-weighted cost per item:
    # Items consuming tight resources are penalized more heavily
    weighted_cost = (tightness[:, None] * weights).sum(axis=0)  # shape (100,)
    
    # Heuristic: prize / weighted cost
    heuristic = total_prize / (weighted_cost + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
