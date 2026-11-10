import numpy as np

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint ratios
    ratios = prizes / weights  # shape (5, 100)
    
    # Soft-min via power mean with negative exponent
    # This penalizes items that are inefficient in any constraint
    p = -4.0
    inv_ratios = ratios ** p  # = ratios^(-4)
    soft_min = (inv_ratios.mean(axis=0)) ** (1.0 / p)  # = (mean(ratios^(-4)))^(-1/4)
    
    # Also compute a slack bonus: items with smaller max weight are more flexible
    max_weight = weights.max(axis=0)  # shape (100,)
    slack = 1.0 / (max_weight + 1e-9)
    
    # Combine: soft_min efficiency * slack bonus
    heuristic = soft_min * slack
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
