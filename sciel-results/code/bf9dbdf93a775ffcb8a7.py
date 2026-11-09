import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute edge usage matrices
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        d_cw = (t - s) % n
        if d_cw != 0:
            for k in range(d_cw):
                e = (s + k) % n
                cw_mat[i, e] = 1.0
            for e in range(n):
                if cw_mat[i, e] == 0.0:
                    ccw_mat[i, e] = 1.0

    # Base load if all demands go CW
    base_load = weights @ cw_mat  # shape (n,)
    
    # Delta vector for flipping demand i from CW to CCW
    # delta[i] = w_i * (ccw_mat[i] - cw_mat[i])
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]
    
    # Gray code enumeration for efficient incremental updates
    # Gray code: g = mask ^ (mask >> 1)
    # Consecutive Gray codes differ by exactly one bit
    # To find which bit changed: diff = g_prev ^ g_cur, bit = index of set bit in diff
    
    best_max_load = np.inf
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Start with mask = 0 (all CW)
    current_load = base_load.copy()
    best_max_load = np.max(current_load)
    best_mask = 0
    
    prev_gray = 0
    for step in range(1, 1 << D):
        if time.time() > deadline:
            break
        
        # Compute current Gray code
        gray = step ^ (step >> 1)
        
        # Find which bit changed
        diff = prev_gray ^ gray
        # diff is a power of 2, so bit_length() - 1 gives the index
        bit = diff.bit_length() - 1
        
        # Determine if bit was set or cleared
        if gray & (1 << bit):
            # Bit was set: add delta (flip from CW to CCW)
            current_load += delta_mat[bit]
        else:
            # Bit was cleared: subtract delta (flip from CCW to CW)
            current_load -= delta_mat[bit]
        
        prev_gray = gray
        
        max_load = np.max(current_load)
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_mask = gray
    
    # Reconstruct the routing vector from the best mask
    out = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if best_mask & (1 << i):
            out[i] = 1.0
        else:
            out[i] = 0.0
            
    return out
