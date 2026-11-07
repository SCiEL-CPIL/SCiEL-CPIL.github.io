"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Strategy: Model as multi-criteria ranking by normalizing each constraint's
    weight independently to [0,1] and computing the heuristic as the product of
    normalized prize and the inverse of the maximum normalized weight across
    all constraints, prioritizing items highly valuable relative to their
    worst-case resource consumption.
    
    Args:
        prizes: (5, 100) array of prize values
        weights: (5, 100) array of weight values
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Ensure we have valid input
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Normalize each constraint's weights to [0, 1]
    # For each constraint c, normalize weights[c, :] to [0, 1]
    w_min = weights.min(axis=1, keepdims=True)
    w_max = weights.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    # Avoid division by zero
    w_range = np.where(w_range < 1e-12, 1.0, w_range)
    normalized_weights = (weights - w_min) / w_range  # shape (5, 100), values in [0, 1]
    
    # Normalize prizes to [0, 1] per constraint as well for consistency
    p_min = prizes.min(axis=1, keepdims=True)
    p_max = prizes.max(axis=1, keepdims=True)
    p_range = p_max - p_min
    p_range = np.where(p_range < 1e-12, 1.0, p_range)
    normalized_prizes = (prizes - p_min) / p_range  # shape (5, 100), values in [0, 1]
    
    # For each item, compute:
    # - Total normalized prize: sum of normalized prizes across constraints
    # - Maximum normalized weight: max of normalized weights across constraints (worst-case)
    total_norm_prize = normalized_prizes.sum(axis=0)  # shape (100,)
    max_norm_weight = normalized_weights.max(axis=0)  # shape (100,)
    
    # Heuristic: product of normalized prize and inverse of max normalized weight
    # Add small epsilon to avoid division by zero
    heuristic = total_norm_prize / (max_norm_weight + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
