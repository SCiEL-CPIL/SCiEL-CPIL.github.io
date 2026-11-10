import numpy as np
import time

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute CW and CCW load vectors for each demand
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        # CW path from s to t: edges s, s+1, ..., t-1 (mod 10)
        if s == t:
            cw_edges = []
        elif s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_vec = np.zeros(n_nodes)
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = np.zeros(n_nodes)
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
    
    cw_arr = np.array(cw_loads)  # shape (15, 10)
    ccw_arr = np.array(ccw_loads)  # shape (15, 10)
    
    # base = sum of all CW loads
    # delta = ccw - cw for each demand
    base = np.sum(cw_arr, axis=0)  # shape (10,)
    deltas = ccw_arr - cw_arr  # shape (15, 10)
    
    start_time = time.time()
    timeout = 600.0
    safety_margin = 5.0
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search over all 2^15 = 32768 configurations
    # Vectorized: for each mask, load = base + sum of deltas[i] where bit i is set
    # Precompute bit vectors for all masks to enable matrix multiplication
    masks = np.arange(1 << n_demands, dtype=np.int32)
    # Create binary matrix: rows are masks, columns are demands
    bits_matrix = ((masks[:, None] >> np.arange(n_demands)) & 1).astype(np.float64)  # (32768, 15)
    
    # Compute all loads at once: (32768, 15) @ (15, 10) -> (32768, 10)
    all_loads = bits_matrix @ deltas + base[None, :]  # (32768, 10)
    
    # Find the mask with minimum maximum load
    max_loads = np.max(all_loads, axis=1)  # (32768,)
    best_idx = np.argmin(max_loads)
    best_max_load = max_loads[best_idx]
    best_mask = int(masks[best_idx])
    
    # Construct result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
            
    return routing
