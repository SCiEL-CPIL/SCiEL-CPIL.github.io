import numpy as np

def heuristics(demands, capacity):
    """
    Compute the heuristic matrix for ACO bin packing.
    
    Args:
        demands: Array of item sizes.
        capacity: Bin capacity.
        
    Returns:
        Matrix H where H[i, j] is the attractiveness of placing item j after item i.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate sum of pairs
    sums = demands[:, None] + demands[None, :]
    
    # Calculate waste (empty space)
    waste = capacity - sums
    
    # Valid if sum <= capacity
    valid = sums <= capacity
    
    # Heuristic: Inverse square of waste, smoothed by epsilon
    # This strongly favors tight fits (low waste)
    eps = 0.5
    # Use where to avoid division by zero or negative bases if logic fails
    # waste can be negative for invalid, but we mask those out.
    # For valid, waste >= 0.
    safe_waste = np.maximum(waste, 0.0)
    
    H = np.zeros((n, n), dtype=float)
    H[valid] = 1.0 / (safe_waste[valid] + eps) ** 2
    H[~valid] = 1e-9
    
    return H
