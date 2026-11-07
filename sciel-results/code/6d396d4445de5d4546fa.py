"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Structural idea from inspiration 1 (simple prize/total_weight ratio scored 1.0)
    combined with multi-criteria normalization from the directive:
    
    1. Normalize each constraint's weights independently to [0, 1] range.
    2. Compute normalized prize (prize / total_prize_sum) for each item.
    3. The heuristic is the product of normalized prize and the inverse of 
       the maximum normalized weight across all constraints.
    
    This prioritizes items that are highly valuable relative to their worst-case
    resource consumption, while maintaining the simplicity that made the 
    total-ratio approach effective.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Ensure positive weights to avoid division issues
    safe_weights = np.maximum(weights, 1e-9)
    
    # Normalize each constraint's weights to [0, 1] range
    # For each constraint c, weight_c_i / max_i(weight_c_i)
    w_max_per_constraint = weights.max(axis=1, keepdims=True)  # shape (5, 1)
    w_max_safe = np.maximum(w_max_per_constraint, 1e-9)
    normalized_weights = weights / w_max_safe  # shape (5, 100), each in [0, 1]
    
    # Maximum normalized weight across all constraints for each item
    # This represents the worst-case resource consumption relative to capacity
    max_norm_weight = normalized_weights.max(axis=0)  # shape (100,)
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Normalize prizes: divide by sum of all prizes to get a relative value
    total_prize_sum = total_prize.sum()
    if total_prize_sum < 1e-9:
        total_prize_sum = 1e-9
    normalized_prize = total_prize / total_prize_sum  # shape (100,)
    
    # Heuristic: product of normalized prize and inverse of max normalized weight
    # Items with high prize and low worst-case resource usage get high scores
    inv_max_norm_weight = 1.0 / (max_norm_weight + 1e-9)
    
    scores = normalized_prize * inv_max_norm_weight
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
