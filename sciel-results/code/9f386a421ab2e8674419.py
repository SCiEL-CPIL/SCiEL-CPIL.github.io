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
    # cw_mask[i, e] is 1 if edge e is in CW path for demand i
    cw_mask = np.zeros((D, n_nodes), dtype=int)
    ccw_mask = np.zeros((D, n_nodes), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            # CW is empty, CCW is all
            ccw_mask[i, :] = 1
        else:
            e = s
            while e != t:
                cw_mask[i, e] = 1
                e = (e + 1) % n_nodes
            # CCW is the rest
            ccw_mask[i, :] = 1 - cw_mask[i, :]
            
    # Weights vector
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # Load if all CW
    # L_cw[e] = sum(weights[i] * cw_mask[i, e])
    L_cw = weights @ cw_mask
    
    # Change in load if we switch demand i to CCW
    # Delta[i, e] = weights[i] * (ccw_mask[i, e] - cw_mask[i, e])
    Delta = (weights[:, None] * (ccw_mask - cw_mask))
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    # Exhaustive Search
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Get binary vector
        # x[i] = 1 if bit i is set (CCW)
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)
        
        # Load = L_cw + sum(x * Delta, axis=0)
        loads = L_cw + np.sum(x[:, None] * Delta, axis=0)
        
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x
    
    return best_routing
