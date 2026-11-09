import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Build masks
    cw_mask = np.zeros((D, n_nodes), dtype=int)
    ccw_mask = np.zeros((D, n_nodes), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            ccw_mask[i, :] = 1
        else:
            e = s
            while e != t:
                cw_mask[i, e] = 1
                e = (e + 1) % n_nodes
            ccw_mask[i, :] = 1 - cw_mask[i, :]
            
    weights = np.array([d[2] for d in demands], dtype=float)
    L_cw = weights @ cw_mask
    Delta = weights[:, None] * (ccw_mask - cw_mask)
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 590
    
    # Gray Code Exhaustive Search
    # We iterate through all 2^D combinations using Gray code
    # to allow incremental updates of the load vector.
    
    # Current state: all CW (x = 0)
    current_loads = L_cw.copy()
    current_x = np.zeros(D, dtype=int)
    
    # Check initial state
    max_load = np.max(current_loads)
    if max_load < best_max_load:
        best_max_load = max_load
        best_routing = current_x.copy()
        
    # Generate Gray code sequence
    # Gray code g = i ^ (i >> 1)
    # We iterate i from 1 to 2^D - 1
    # The difference between gray(i) and gray(i-1) is a single bit flip.
    
    prev_gray = 0
    for i in range(1, 1 << D):
        if time.time() > deadline:
            break
            
        gray = i ^ (i >> 1)
        
        # Find which bit changed
        diff = gray ^ prev_gray
        # Index of the changed bit
        bit_idx = int(np.log2(diff))
        
        # Determine if the bit turned on (0->1) or off (1->0)
        # If the bit is set in the new gray code, it turned on.
        # If it's clear, it turned off.
        if gray & diff:
            # Bit turned on (0 -> 1)
            # Add Delta[bit_idx]
            current_loads += Delta[bit_idx]
            current_x[bit_idx] = 1
        else:
            # Bit turned off (1 -> 0)
            # Subtract Delta[bit_idx]
            current_loads -= Delta[bit_idx]
            current_x[bit_idx] = 0
            
        max_load = np.max(current_loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_x.copy()
            
        prev_gray = gray
        
    return best_routing
