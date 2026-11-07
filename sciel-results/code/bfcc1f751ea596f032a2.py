import numpy as np

def heuristics(prizes, weights):
    """
    Returns a heuristic array of shape (100,) for the MCKP.
    Uses a weighted blend of sum-ratio, geo-ratio, and soft-min-ratio heuristics.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Clip values to prevent division by zero and log of zero
    eps = 1e-9
    p = np.clip(prizes, eps, None)
    w = np.clip(weights, eps, None)
    
    # Sum-ratio heuristic: total prize / total weight
    sum_ratio = p.sum(axis=0) / w.sum(axis=0)
    
    # Per-constraint ratios: shape (5, 100)
    ratios = p / w
    
    # Geometric mean of ratios
    # log(geo_mean) = mean(log(ratios))
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)
    geo_ratio = np.exp(mean_log_ratio)
    
    # Soft-min ratio: smooth approximation of min over constraints
    # Using log-sum-exp trick: soft_min(x) = -log(sum(exp(-x))) for large beta
    # We want items that are good across ALL constraints, so penalize the worst constraint
    # soft_min of ratios: use a power mean with low exponent (p=1 is geo mean, p->0 is min)
    # Actually, let's use a smooth minimum: -log(sum(exp(-beta * ratios)))/beta for large beta
    # But this can overflow. Let's use a different approach.
    # Power mean with exponent -1: harmonic mean (closer to min than geo mean)
    # Or use: exp(mean(log(ratios^k))) = geo_mean(ratios^k) = geo_mean(ratios)^k
    # For k < 1, this emphasizes smaller values (more like min)
    # Let's try k = 0.5: sqrt of geo mean = exp(0.5 * mean(log(ratios)))
    soft_min_ratio = np.exp(0.5 * mean_log_ratio)
    
    # Normalize all heuristics to have similar scale
    # Use min-max normalization to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > eps:
            return (arr - min_val) / (max_val - min_val)
        return np.ones_like(arr) / arr.size
    
    sum_ratio_norm = normalize(sum_ratio)
    geo_ratio_norm = normalize(geo_ratio)
    soft_min_norm = normalize(soft_min_ratio)
    
    # Blend: weight more toward sum-ratio (higher individual score) and soft-min
    # (captures multi-constraint structure better than pure geo mean)
    heuristic = 0.5 * sum_ratio_norm + 0.25 * geo_ratio_norm + 0.25 * soft_min_norm
    
    # Scale back up to avoid being too small (ACO uses pheromone * heuristic)
    heuristic = heuristic + 1e-9
    
    # Ensure all values are positive finite
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
