import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Feasibility mask
    feasible = pair_sum <= capacity
    
    # Calculate heuristic
    # We want to minimize waste: waste = capacity - pair_sum
    # Heuristic = 1 / (waste + 1)
    # For infeasible, set to 1e-9
    
    # Avoid division by zero or negative numbers
    waste = capacity - pair_sum
    # Only compute where feasible
    # For infeasible, waste is negative, so 1/(waste+1) might be weird.
    # We explicitly mask.
    
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Compute for feasible pairs
    # waste is >= 0 for feasible
    # Add 1 to avoid div by 0 and scale
    valid_waste = waste[feasible] + 1.0
    H[feasible] = 1.0 / valid_waste
    
    # Check for non-finite or non-positive
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Time guard (unlikely to be needed)
    if time.time() - start_time > 320:
         # Return a safe fallback if somehow slow (shouldn't happen)
         return np.ones((n, n)) * 1e-9

    return H
