import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0

    # Precompute CW masks
    cw_mask = np.zeros((D, n), dtype=np.int8)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
            
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # B: Load if all demands are routed CW
    B = (weights[:, None] * cw_mask).sum(axis=0)
    
    # D_vec[d]: Difference in load vector if demand d is switched from CW to CCW
    ones = np.ones(n, dtype=np.int8)
    D_vec = weights[:, None] * (ones[None, :] - 2 * cw_mask)
    
    # Exhaustive search (fast for 2^15)
    B_list = B.tolist()
    D_vec_list = D_vec.tolist()
    
    best_max_load = float('inf')
    best_gray = 0
    current_sum = [0.0] * n
    prev_gray = 0
    
    # 2^15 is 32768 iterations. This is very fast.
    # We can do it all in one go.
    for g in range(2**D):
        if time.time() > deadline - 1.0:
            break
            
        gray = g ^ (g >> 1)
        
        if g > 0:
            diff = gray ^ prev_gray
            idx = diff.bit_length() - 1
            
            if (gray >> idx) & 1:
                dv = D_vec_list[idx]
                for e in range(n):
                    current_sum[e] += dv[e]
            else:
                dv = D_vec_list[idx]
                for e in range(n):
                    current_sum[e] -= dv[e]
                    
        prev_gray = gray
        
        max_load = 0.0
        for e in range(n):
            load = B_list[e] + current_sum[e]
            if load > max_load:
                max_load = load
                
        if max_load < best_max_load:
            best_max_load = max_load
            best_gray = gray
            
    routing = np.zeros(D, dtype=float)
    for d in range(D):
        if (best_gray >> d) & 1:
            routing[d] = 1.0
        else:
            routing[d] = 0.0
            
    return routing
