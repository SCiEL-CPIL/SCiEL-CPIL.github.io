import numpy as np

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # Calculate sum of all pairs
    pair_sums = demands[:, None] + demands[None, :]
    
    # Initialize heuristic matrix with very small value for invalid pairs
    eta = np.full((n, n), 1e-9)
    
    # Find valid pairs (sum <= capacity)
    valid_mask = pair_sums <= capacity
    
    # Calculate waste for valid pairs
    # Waste = capacity - sum
    # We want to maximize attractiveness for low waste.
    # Attractiveness ~ 1 / (waste^2 + 1)
    # This avoids division by zero and strongly penalizes large waste.
    
    waste = capacity - pair_sums
    # For invalid pairs, waste > 0, but we mask them out.
    # We only compute for valid ones.
    
    # To avoid modifying the whole array with errors, compute on masked array
    # Or just compute everywhere and mask later.
    # waste is positive for valid, negative for invalid (sum > cap).
    # But we only care about valid.
    
    # Safe calculation
    # We want eta[i,j] = 1 / (waste[i,j]^2 + 1) if valid, else 1e-9
    
    # Calculate the metric for all, then mask
    # Note: For invalid pairs, waste is negative. waste^2 is positive.
    # But we want them to be 1e-9.
    
    metric = 1.0 / (waste**2 + 1.0)
    
    # Apply mask
    eta[valid_mask] = metric[valid_mask]
    
    return eta
