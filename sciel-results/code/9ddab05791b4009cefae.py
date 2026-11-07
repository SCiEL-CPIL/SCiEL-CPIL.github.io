"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Validity mask: item j can follow item i if their sum <= capacity
    valid = pair_sum <= capacity
    
    # Calculate waste for valid pairs
    waste = capacity - pair_sum
    waste = np.where(valid, waste, 0)
    
    # Fit heuristic: prefer less waste with sharper discrimination.
    # Using 1/(1+waste)^3 gives even stronger penalty for larger waste,
    # making tight fits (small waste) much more attractive than loose fits.
    fit = 1.0 / (1.0 + waste) ** 3
    
    # Scarcity heuristic: prefer items that have fewer compatible partners
    # compat_count[j] is the number of items i such that d_i + d_j <= capacity
    compat_count = valid.sum(axis=0).astype(float)
    
    # Use inverse cube for stronger scarcity signal
    scarcity = 1.0 / (compat_count ** 3)
    
    # Forward flexibility: after placing j after i, how many items k can still fit?
    # remaining_capacity_after = capacity - (d_i + d_j)
    # k fits if d_k <= remaining_capacity_after
    # This rewards placements that leave more room for subsequent items
    rem_cap = capacity - pair_sum  # shape (n, n)
    rem_cap = np.where(valid, rem_cap, 0)
    
    # For each (i, j) pair, count how many items k satisfy d_k <= rem_cap[i,j]
    # demands is sorted? No, but we can use np.searchsorted on sorted demands
    sorted_demands = np.sort(demands)
    # For each (i,j), count items with demand <= rem_cap[i,j]
    # Use searchsorted to find how many sorted demands are <= rem_cap[i,j]
    # rem_cap shape (n, n), sorted_demands shape (n,)
    # np.searchsorted(sorted_demands, rem_cap, side='right') gives count
    flex_counts = np.searchsorted(sorted_demands, rem_cap, side='right')
    flex_counts = flex_counts.astype(float)
    
    # Normalize flexibility: higher is better (more options remaining)
    # Use a smooth function: 1 + flex_counts to avoid division by zero
    flexibility = 1.0 + flex_counts
    
    # Combine heuristics:
    # eta[i, j] = fit[i,j] * scarcity[j] * flexibility[i,j]
    eta = fit * scarcity[None, :] * flexibility
    
    # Set invalid entries to 1e-9
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Deadline guard
    if time.time() - start_time > 250:
         # Fallback to seed-like heuristic if somehow slow (unlikely)
         pair = demands[:, None] + demands[None, :]
         return np.where(pair <= capacity, pair / capacity, 1e-9)
         
    return eta
# EVOLVE-BLOCK-END
