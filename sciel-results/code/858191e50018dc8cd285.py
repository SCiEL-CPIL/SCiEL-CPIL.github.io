import numpy as np

def heuristics(prizes, weights):
    """
    Returns a heuristic array of shape (100,) for the MCKP.
    Uses the geometric mean of the per-constraint prize-to-weight ratios.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Clip values to prevent division by zero and log of zero
    eps = 1e-9
    p = np.clip(prizes, eps, None)
    w = np.clip(weights, eps, None)
    
    # Per-constraint ratios: shape (5, 100)
    ratios = p / w
    
    # Geometric mean of ratios
    # log(geo_mean) = mean(log(ratios))
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)
    
    # Convert back to linear scale
    heuristic = np.exp(mean_log_ratio)
    
    # Ensure all values are positive finite
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
