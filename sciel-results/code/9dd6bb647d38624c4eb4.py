"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
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
    
    # Average weight across constraints (overall resource usage)
    avg_weight = weights.mean(axis=0)
    
    # Heuristic: reward high total prize, penalize bottleneck and overall weight
    # Multiplicative penalty: items heavy in any constraint are strongly discouraged
    h = total_prize / ((1.0 + max_weight) * (1.0 + avg_weight) + 1e-9)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
