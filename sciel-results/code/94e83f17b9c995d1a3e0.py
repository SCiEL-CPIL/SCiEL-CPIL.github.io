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
    
    # Use Geometric Mean of per-constraint efficiency ratios
    # This balances value against the multiplicative cost of consuming resources
    # across all dimensions simultaneously. It penalizes items with high variance
    # in resource consumption (inefficient in some constraints) more than
    # arithmetic mean, and less aggressively than min for single bad constraints.
    
    # Geometric mean = exp(mean(log(ratios)))
    # To handle numerical stability and potential zeros/negatives (though weights/prizes are positive in [0,1]),
    # we work in log space.
    
    # ratios are positive because prizes, weights > 0 (uniform [0,1] implies > 0 almost surely, 
    # but safe_weights ensures no division by zero. If prizes are 0, log is -inf).
    # We clip ratios to a small positive value to avoid log(0) = -inf.
    safe_ratios = np.maximum(ratios, eps)
    
    log_ratios = np.log(safe_ratios)
    
    # Geometric mean is exp(mean(log_ratios, axis=0))
    mean_log_ratios = np.mean(log_ratios, axis=0)
    geom_mean = np.exp(mean_log_ratios)
    
    # Ensure all values are positive and finite
    # Clip to avoid extreme values
    geom_mean = np.clip(geom_mean, eps, 1e6)
    
    # Replace any non-finite values with a small positive number
    geom_mean = np.where(np.isfinite(geom_mean), geom_mean, eps)
    
    # Normalize the heuristic to have a more consistent scale
    # This helps the ACO algorithm balance pheromone and heuristic contributions
    # by ensuring the heuristic values are in a reasonable range
    mean_geom_mean = np.mean(geom_mean)
    if mean_geom_mean > 0:
        geom_mean = geom_mean / mean_geom_mean
    
    soft_min = geom_mean
    
    # Ensure the output has the correct shape (100,)
    return soft_min
# EVOLVE-BLOCK-END
