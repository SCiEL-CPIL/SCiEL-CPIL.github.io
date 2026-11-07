"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Uses normalized weights per constraint to capture relative tightness,
    then combines geometric mean and soft-min of efficiency ratios.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    eps = 1e-9
    capacity = 25.0
    
    # Compute residual capacity for each constraint
    # This represents how much "room" remains after accounting for all items' weights
    # A constraint with small total weight has large residual -> more abundant resource
    total_weights = np.sum(weights, axis=1)  # shape (5,)
    residual_capacity = capacity - total_weights  # shape (5,)
    # Ensure residual is positive (if all items fit, residual > 0; clip for safety)
    residual_capacity = np.maximum(residual_capacity, eps)
    
    # Per-constraint efficiency: prize / weight
    safe_weights = np.maximum(weights, eps)
    base_ratios = prizes / safe_weights  # shape (5, 100)
    
    # Scale each constraint's efficiency by its residual capacity
    # Items in constraints with more remaining capacity get boosted efficiency
    # This makes the heuristic "greedy feasibility-aware" - prioritize using abundant resources
    residual_scaled = residual_capacity[:, np.newaxis]  # shape (5, 1) for broadcasting
    scaled_ratios = base_ratios * residual_scaled  # shape (5, 100)
    
    # Geometric mean across constraints
    log_efficiency = np.log(np.maximum(scaled_ratios, eps))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Soft-Min to capture worst-case bottleneck
    beta = 2.0
    v = -beta * scaled_ratios
    v_max = np.max(v, axis=0, keepdims=True)
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    log_sum_exp = v_max + np.log(sum_exp + eps)
    soft_min_ratios = -1.0 / beta * log_sum_exp
    
    # Combine geometric mean and soft-min
    scores = 0.5 * geo_mean_efficiency + 0.5 * soft_min_ratios
    
    # Ensure all values are positive
    scores = np.maximum(scores, eps)
    
    return scores
# EVOLVE-BLOCK-END
