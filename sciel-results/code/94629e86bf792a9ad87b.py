import numpy as np

def heuristics(prizes, weights):
    # Ensure prizes is 1D
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    
    # Use sum of weights as denominator - captures total resource consumption
    # across all constraints, which is more informative than just the max
    w_sum = np.sum(weights, axis=0)
    
    # Add epsilon to avoid division by zero
    heuristic = prizes / (w_sum + 1e-9)
    
    # Ensure positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
