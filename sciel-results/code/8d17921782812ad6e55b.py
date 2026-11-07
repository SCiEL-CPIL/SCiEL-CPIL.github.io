import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Args:
        demands: array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        matrix: n x n matrix where entry (i, j) is the attractiveness of 
                placing item j after item i.
    """
    n = len(demands)
    d = np.array(demands)
    
    # Vectorized calculation of pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Initialize heuristic matrix with a very small positive value for non-fitting pairs
    H = np.full((n, n), 1e-9)
    
    # Mask for fitting pairs
    fit_mask = pair_sums <= capacity
    
    if not np.any(fit_mask):
        return H

    # Calculate waste for fitting pairs
    waste = capacity - pair_sums
    
    # Heuristic components:
    # 1. Fill Ratio: (d_i + d_j) / capacity. Rewards larger combined sizes.
    # 2. Waste Penalty: 1 / (1 + waste)^alpha. Rewards tight fits.
    
    # Tuning parameters
    # alpha controls the strength of the waste penalty.
    # gamma controls the strength of the size preference.
    
    alpha = 0.5
    gamma = 1.0
    
    # Calculate for fitting pairs only
    sums_fit = pair_sums[fit_mask]
    waste_fit = waste[fit_mask]
    
    fill_ratio = sums_fit / capacity
    waste_penalty = 1.0 / np.power(1.0 + waste_fit, alpha)
    
    H_fit = np.power(fill_ratio, gamma) * waste_penalty
    
    H[fit_mask] = H_fit
    
    return H
