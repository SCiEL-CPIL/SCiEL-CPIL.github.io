"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Maximum weight across constraints (bottleneck constraint)
    max_weight = weights.max(axis=0)
    
    # Sum of weights across constraints (overall resource usage)
    total_weight = weights.sum(axis=0)
    
    # Combine bottleneck and total weight into a single cost measure
    # Items that are heavy in any single constraint are more costly
    # The max weight captures the binding constraint, sum captures overall usage
    combined_cost = 0.5 * max_weight + 0.5 * total_weight
    
    # Heuristic: prize density with respect to combined cost
    # High total prize and low combined cost -> High attractiveness
    h = total_prize / (combined_cost + 1e-9)
    
    # Ensure positivity as required by the task description
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
