import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing configurations.
    This is exact and fast enough for N=15 demands.
    """
    start_time = time.time()
    deadline = start_time + 600.0

    n = 10
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
        (5, 2, 2),
    ]
    d = len(demands)
    
    # Pre-compute CW and CCW edge usage matrices (d, n)
    cw = np.zeros((d, n), dtype=np.int64)
    ccw = np.zeros((d, n), dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)
    
    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]
        
    # Base load if all demands go CW
    base_load = np.sum(w[:, None] * cw, axis=0) # shape (n,)
    
    # Delta load if demand i switches from CW to CCW
    # delta[i] = w[i] * (ccw[i] - cw[i])
    delta = w[:, None] * (ccw - cw) # shape (d, n)
    
    best_max_load = np.inf
    best_config = np.zeros(d, dtype=np.int64)
    
    # Iterate over all 2^15 configurations
    # x_i = 1 means CCW, 0 means CW
    # We iterate through integers 0 to 2^d - 1
    # Bit i of the integer represents x_i
    
    # Optimization: We can use a Gray code to update the load incrementally,
    # but for 2^15, direct calculation is also fast.
    # However, updating incrementally is O(1) per step vs O(d*n) per step.
    # Let's use Gray code for speed.
    
    current_load = base_load.copy()
    current_config = np.zeros(d, dtype=np.int64)
    
    for mask in range(1 << d):
        if time.time() > deadline:
            break
            
        # If mask is 0, we are at base_load (all CW)
        # We need to update current_load from the previous mask to the current mask.
        # Using Gray code: g = mask ^ (mask >> 1)
        # The difference between consecutive Gray codes is one bit.
        
        if mask == 0:
            current_max = np.max(current_load)
            if current_max < best_max_load:
                best_max_load = current_max
                best_config = current_config.copy()
            continue
            
        # Calculate Gray code for current mask and previous mask
        gray_curr = mask ^ (mask >> 1)
        gray_prev = (mask - 1) ^ ((mask - 1) >> 1)
        
        # Find the bit that changed
        diff = gray_curr ^ gray_prev
        bit_idx = diff.bit_length() - 1
        
        # Determine if we turned ON (CW->CCW) or OFF (CCW->CW)
        # If bit in gray_curr is 1, it turned ON (was 0 in gray_prev)
        if gray_curr & diff:
            # Turned ON: x_i became 1. Add delta[i]
            current_load += delta[bit_idx]
            current_config[bit_idx] = 1
        else:
            # Turned OFF: x_i became 0. Subtract delta[i]
            current_load -= delta[bit_idx]
            current_config[bit_idx] = 0
            
        current_max = np.max(current_load)
        if current_max < best_max_load:
            best_max_load = current_max
            best_config = current_config.copy()

    # best_config[i] is 1 if CCW, 0 if CW
    # Output format: >= 0.5 for CCW, < 0.5 for CW
    return best_config.astype(np.float64)
