"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pairwise sums and waste
    # waste[i, j] is the remaining capacity if item j is placed after item i
    waste = capacity - demands[:, None] - demands[None, :]
    
    # Validity mask
    valid = waste >= 0
    
    # Base fit score: prefer less waste
    # Use 1/(1+waste) for a smooth decay
    fit_score = 1.0 / (1.0 + np.where(valid, waste, 1.0))
    
    # Calculate flexibility score:
    # For each pair (i, j), how many items k can fit in the residual waste[i, j]?
    # This penalizes moves that create "dead space" (residual too small for any remaining item)
    
    # Vectorized calculation of fits_count
    # waste is (n, n), demands is (n,)
    # We want to compare waste[i, j] with every demands[k]
    # Result shape: (n, n, n)
    # This is 120^3 = 1.7M operations, which is fast.
    
    # To save memory/time if n were larger, we could sort and use searchsorted, 
    # but for n=120, direct comparison is fine.
    
    # Broadcast waste to (n, n, 1) and demands to (1, 1, n)
    waste_expanded = waste[:, :, None]
    demands_expanded = demands[None, None, :]
    
    # Boolean mask: True if demands[k] <= waste[i, j]
    fits_mask = demands_expanded <= waste_expanded
    
    # Count how many items fit in the residual space for each pair (i, j)
    fits_count = fits_mask.sum(axis=2)
    
    # Flexibility score:
    # If fits_count is 0, the residual space is useless for any item.
    # We want to penalize this strongly.
    # If fits_count is high, the residual space is flexible.
    # Let's use (1 + fits_count) so that 0 fits gives a score of 1, 
    # but we can also use a power to increase discrimination.
    # Let's try: flex_score = 1.0 + fits_count
    # Or maybe: flex_score = (1.0 + fits_count) ** 0.5 to dampen the effect?
    # Given the "bottleneck" directive, we want to strongly penalize dead ends.
    # Let's stick with linear or slightly super-linear.
    
    flex_score = 1.0 + fits_count
    
    # Combine heuristics
    # eta[i, j] = attractiveness of placing j after i
    eta = fit_score * flex_score
    
    # Set invalid entries to 1e-9
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Simple deadline guard
    if time.time() - start_time > 250:
         # Fallback
         pair = demands[:, None] + demands[None, :]
         return np.where(pair <= capacity, 1.0 / (1.0 + (capacity - pair)), 1e-9)
         
    return eta
# EVOLVE-BLOCK-END
