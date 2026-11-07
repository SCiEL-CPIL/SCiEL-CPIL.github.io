"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    s = np.array(demands)
    
    # Compute sum of all pairs
    sums = s[:, None] + s[None, :]
    
    # Compute ideal complement for each i
    # Ideal size for j given i is capacity - s_i
    ideal = capacity - s[:, None] # Shape (n, 1)
    
    # Difference between actual j size and ideal size
    # s[None, :] is shape (1, n)
    diff = s[None, :] - ideal
    
    # Gaussian heuristic with tuned sigma
    sigma = 8.0
    exponent = - (diff ** 2) / (2 * sigma ** 2)
    H = np.exp(exponent)
    
    # Add fill-ratio bonus: reward pairs that use more of the bin capacity
    # fill_ratio = sums / capacity, higher is better
    fill_ratio = sums / capacity
    fill_bonus = fill_ratio ** 2
    
    # Combine: primary Gaussian + secondary fill bonus
    H = H * (1.0 + 0.5 * fill_bonus)
    
    # Mask infeasible pairs
    # If sum > capacity, set to 0 (will be clamped to 1e-9 by evaluator)
    H[sums > capacity] = 0.0
    
    # Ensure minimum positive values for feasible pairs
    feasible_mask = sums <= capacity
    H[feasible_mask] = np.maximum(H[feasible_mask], 1e-6)
    
    return H
# EVOLVE-BLOCK-END
