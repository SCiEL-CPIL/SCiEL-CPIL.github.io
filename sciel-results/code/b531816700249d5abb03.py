import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search using Gray code 
    optimization to efficiently minimize the maximum edge load.
    """
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
    
    # Precompute edge sets for CW and CCW
    # CW: s, s+1, ..., t-1
    # CCW: all other edges
    
    W_cw = np.zeros((n_demands, n_nodes))
    W_ccw = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges
        current = s
        while current != t:
            W_cw[i, current] = w
            current = (current + 1) % n_nodes
        
        # CCW edges
        all_edges = np.ones(n_nodes)
        cw_mask = (W_cw[i] > 0).astype(int)
        ccw_mask = 1 - cw_mask
        W_ccw[i] = ccw_mask * w
        
    # L_base is the load if all demands go CW
    L_base = np.sum(W_cw, axis=0)
    
    # D[i] is the change in load if demand i switches from CW to CCW
    # L = L_base + sum(D[i] for i in mask)
    D = W_ccw - W_cw
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Use Gray code to iterate through all masks efficiently
    # Gray code g = m ^ (m >> 1)
    # To get the bit that changed between consecutive Gray codes, 
    # we can look at the difference between g_prev and g_curr.
    
    prev_gray = 0
    current_load = L_base.copy()
    
    # We iterate m from 0 to 2^n - 1
    # gray(m) = m ^ (m >> 1)
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    for m in range(1 << n_demands):
        if time.time() > deadline:
            break
            
        gray = m ^ (m >> 1)
        
        if m > 0:
            # Find the bit that changed
            diff = prev_gray ^ gray
            # diff is a power of 2, so the index is log2(diff)
            bit_index = int(np.log2(diff))
            
            # Determine if the bit was set in the previous mask or current mask
            # If prev_gray had the bit set, we are switching from CCW to CW (subtract D)
            # If prev_gray did not have the bit set, we are switching from CW to CCW (add D)
            if prev_gray & diff:
                current_load -= D[bit_index]
            else:
                current_load += D[bit_index]
                
        prev_gray = gray
        
        current_max_load = np.max(current_load)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = m
            
    # Construct the result array from best_mask
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
