"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Calculate total prize for each item
    total_prize = prizes.sum(axis=0)
    
    # Calculate the maximum weight consumption for each item across all constraints
    # This represents the "bottleneck" resource usage of the item
    max_weight = weights.max(axis=0)
    
    # Compute the ratio of total prize to bottleneck weight
    # This heuristic favors items that provide high value without excessively
    # consuming any single resource
    eps = 1e-9
    score = total_prize / (max_weight + eps)
    
    # Ensure all values are positive and finite
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
