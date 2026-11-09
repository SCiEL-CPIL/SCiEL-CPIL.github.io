"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # demands is a 1D array of floats
    # Vectorized calculation for efficiency
    
    # Pair sums: demands[i] + demands[j]
    # We want to check if they fit in capacity
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid pairs are those where sum <= capacity
    valid = pair_sums <= capacity
    
    # Remaining space for valid pairs
    rem = capacity - pair_sums
    
    # Initialize heuristic matrix
    heur = np.full((n, n), 1e-9, dtype=np.float64)
    
    if not np.any(valid):
        return heur
        
    # Strategy: Favor small items (1/d_j) that also fit tightly (1/(rem+1))
    # d_j is demands[j]
    # We need to be careful with d_j being 0? Demands are 20-99, so safe.
    
    # d_j broadcasted to rows
    d_j = demands[None, :]
    
    # Calculate heuristic for valid pairs
    # term1: 1 / d_j  (Favors smaller items to fill gaps)
    # term2: 1 / (rem + 1) (Favors tighter packs)
    # Combined: term1 * term2
    
    # To avoid division by zero in term2 (rem can be 0), we use rem + 1.
    # Since demands are integers, min rem is 0.
    
    # Compute only for valid entries to save time/memory if n is large, 
    # but n=120 is small, so full matrix is fine.
    
    # We must handle the case where valid is False. 
    # We create a temporary array for the calculation.
    
    # rem_valid = rem[valid]
    # d_j_valid = d_j[valid] # This is tricky with boolean indexing on 2D broadcast
    
    # Easier way:
    # Create a safe_rem array where invalid entries have a large value (so 1/large is small)
    # Create a safe_d_j array where invalid entries have a large value
    
    # Or just compute on the whole matrix and mask.
    
    # d_j_safe = np.where(d_j > 0, d_j, 1.0) # d_j is always >= 20
    # rem_safe = np.where(valid, rem + 1.0, 1.0) # If invalid, we don't care about the value, we will mask later
    
    # However, division by zero or inf might occur if not careful.
    # Let's compute on valid indices only for clarity and safety.
    
    # Get indices of valid pairs
    idx_i, idx_j = np.where(valid)
    
    # Extract values
    rem_vals = rem[idx_i, idx_j]
    d_j_vals = demands[idx_j]
    
    # Calculate heuristic
    # h = (1.0 / d_j_vals) * (1.0 / (rem_vals + 1.0))
    
    # This heuristic strongly prefers:
    # 1. Small items j (large 1/d_j)
    # 2. Tight fits (large 1/(rem+1))
    
    heur_vals = (1.0 / d_j_vals) * (1.0 / (rem_vals + 1.0))
    
    heur[idx_i, idx_j] = heur_vals
    
    # Ensure no non-positive values (should be positive given demands > 0 and rem >= 0)
    # heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
