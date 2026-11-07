import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of MCKP.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0,1]
        weights: array of shape (5, 100), uniform in [0,1]
    
    Returns:
        array of shape (100,) with one positive finite score per item
    """
    eps = 1e-9
    
    # Per-constraint efficiency: prize / weight
    ratios = prizes / (weights + eps)
    
    # Harmonic mean across constraints (penalizes tight constraints)
    inv_ratios = 1.0 / ratios
    sum_inv = np.sum(inv_ratios, axis=0)
    harmonic = 5.0 / sum_inv
    
    # Minimum ratio (bottleneck constraint)
    min_ratio = np.min(ratios, axis=0)
    
    # Geometric mean (alternative balance)
    log_ratios = np.log(ratios + eps)
    geo_mean = np.exp(np.mean(log_ratios, axis=0))
    
    # Combine: emphasize harmonic (best measured) with min ratio and geo mean
    # Weighted combination to capture both balance and bottleneck
    combined = 0.5 * harmonic + 0.3 * min_ratio + 0.2 * geo_mean
    
    # Ensure positive
    combined = np.maximum(combined, eps)
    
    # Amplify differences with power transform
    # Higher power = sharper preference for best items
    power = 1.5
    eta = np.power(combined, power)
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta
