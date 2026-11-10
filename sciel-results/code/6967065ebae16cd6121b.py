import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    Uses an inverse gap heuristic: higher attractiveness for pairs that
    leave less residual space in the bin.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pair sums
    # Shape: (n, n)
    pair_sums = demands[:, None] + demands[None, :]
    
    # Calculate residual space
    residuals = capacity - pair_sums
    
    # Feasibility mask: pair fits in bin
    feasible = residuals >= 0
    
    # Initialize heuristic matrix with a very small value for infeasible pairs
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Calculate heuristic for feasible pairs
    # We want to reward small residuals.
    # H = 1 / (residual + epsilon)
    # Using a small epsilon to avoid division by zero and to cap the max value.
    # If residual is 0, H = 1/epsilon.
    # If residual is large, H is small.
    
    # Choose epsilon. 
    # If epsilon is too small, values can be huge, potentially causing numerical issues 
    # or dominating pheromones completely. 
    # However, ACO usually handles this with scaling or just large probabilities.
    # Let's use epsilon = 1.0 to keep values in a reasonable range (1 to 1/150).
    # Or epsilon = 0.1 for stronger preference.
    # Given the "Inverse gap" knowledge, a strong penalty for space is good.
    # Let's try epsilon = 1.0 first for stability, or maybe 0.5.
    # Actually, the relative order is what matters most.
    
    epsilon = 1.0
    safe_residuals = residuals + epsilon
    
    # Avoid division by zero just in case (though safe_residuals >= 1)
    # H[feasible] = 1.0 / safe_residuals[feasible]
    
    # To be safe and efficient:
    with np.errstate(divide='ignore', invalid='ignore'):
        H_vals = 1.0 / safe_residuals
        H_vals[~feasible] = 1e-9
        
    # Ensure all values are positive and finite
    # H_vals are positive.
    
    return H_vals
