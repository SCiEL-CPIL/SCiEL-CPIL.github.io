"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Uses a soft-min of per-constraint efficiency ratios to identify the bottleneck
    constraint efficiency for each item, which better approximates the true bottleneck
    than geometric mean or arithmetic sum.
    
    Args:
        prizes: array of shape (5, 100) with prize values per constraint and item
        weights: array of shape (5, 100) with weight values per constraint and item
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # prizes: (5, 100), weights: (5, 100)
    eps = 1e-9
    
    # Calculate per-constraint efficiency ratios
    # Shape: (5, 100)
    safe_weights = np.maximum(weights, eps)
    ratios = prizes / safe_weights
    
    # Use a Soft-Min approximation to identify the bottleneck constraint efficiency
    # SoftMin(x) = -1/beta * log(sum(exp(-beta * x)))
    # This smooths the minimum of the ratios, penalizing items that have
    # very low efficiency in any single constraint.
    # A larger beta makes the soft-min closer to the true min (sharper bottleneck detection)
    # A smaller beta makes it smoother (more forgiving of one bad constraint)
    # Based on inspiration 1, beta=2.0 scored 1.0812, so we use that value
    beta = 2.0
    
    # For numerical stability, subtract the max of (-beta * ratios) before exp
    v = -beta * ratios
    v_max = np.max(v, axis=0, keepdims=True)
    
    # sum(exp(v - v_max))
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    
    # soft_min = -1/beta * (v_max + log(sum_exp))
    # Note: v_max is shape (1, 100), sum_exp is shape (100,)
    soft_min = -1.0 / beta * (v_max[0] + np.log(sum_exp))
    
    # Ensure all values are positive and finite
    # Clip to avoid extreme values
    soft_min = np.clip(soft_min, eps, 1e6)
    
    # Replace any non-finite values with a small positive number
    soft_min = np.where(np.isfinite(soft_min), soft_min, eps)
    
    # Normalize the heuristic to have a more consistent scale
    # This helps the ACO algorithm balance pheromone and heuristic contributions
    # by ensuring the heuristic values are in a reasonable range
    mean_soft_min = np.mean(soft_min)
    if mean_soft_min > 0:
        soft_min = soft_min / mean_soft_min
    
    # Ensure the output has the correct shape (100,)
    return soft_min
# EVOLVE-BLOCK-END
