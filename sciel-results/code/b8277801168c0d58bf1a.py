"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = 300.0 # Safety margin
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    if n == 0:
        return np.empty((0, 0))
        
    # 1. Compute FFD solution
    # Sort indices by demand descending
    sorted_indices = np.argsort(demands)[::-1]
    
    bins = []
    remaining_cap = []
    
    for idx in sorted_indices:
        if time.time() > deadline - 5:
             # Fallback to simple heuristic if time is tight (unlikely)
             pair_sum = demands[:, None] + demands[None, :]
             fits = pair_sum <= capacity
             eta = np.where(fits, pair_sum / capacity, 1e-9)
             return eta
             
        size = demands[idx]
        placed = False
        for k in range(len(bins)):
            if remaining_cap[k] >= size:
                bins[k].append(idx)
                remaining_cap[k] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining_cap.append(capacity - size)
            
    # 2. Build FFD pair matrix
    ffd_pair = np.zeros((n, n), dtype=bool)
    for b in bins:
        for u in b:
            for v in b:
                if u != v:
                    ffd_pair[u, v] = True
                    ffd_pair[v, u] = True
                    
    # 3. Build Heuristic Matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    eta = np.full((n, n), 1e-9)
    base_val = pair_sum / capacity
    
    eta[fits] = base_val[fits]
    
    # Boost FFD pairs
    boost_factor = 10.0
    eta[ffd_pair] = boost_factor * base_val[ffd_pair]
    
    return eta
# EVOLVE-BLOCK-END
