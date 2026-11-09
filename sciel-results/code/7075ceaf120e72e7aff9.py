import numpy as np
import time

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute CW and CCW edge usage matrices
    # cw_vec[i, j] = 1 if demand i uses edge j clockwise
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
            
    # ccw_vec[i, j] = 1 if demand i uses edge j counter-clockwise
    # Since paths are complementary on a ring:
    ccw_vec = 1 - cw_vec
    
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    
    # Pre-compute Base Load (all CW) and Delta vectors
    # Base_k = sum(w_i * cw_vec[i, k])
    base_load = cw_vec.T @ weights
    
    # Delta_i_k = w_i * (ccw_vec[i, k] - cw_vec[i, k])
    # If we switch demand i from CW to CCW, the load changes by this amount.
    # L = Base + sum(Delta_i for i in S_ccw)
    delta_vecs = (ccw_vec - cw_vec).T * weights[:, np.newaxis]
    # delta_vecs is shape (10, 15). delta_vecs[k, i] is the change in load of edge k if demand i is routed CCW.
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Iterate through all 2^15 masks
    # We can optimize by noting that the problem is small.
    # A simple loop is fine.
    
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
            
        # Calculate load for this mask
        # L = base_load + sum(delta_vecs[:, i] for i where bit i is set)
        
        # Efficient way:
        # Extract bits
        # Since n_demands is 15, we can just loop or use numpy.
        # Looping 15 times and summing 15 vectors of length 10 is 150 ops.
        # 32768 * 150 = ~5 million ops. Very fast.
        
        current_load = base_load.copy()
        
        # Unroll the bit extraction for speed?
        # Or just use numpy bit operations.
        # Let's try a simple loop over bits, it's likely faster than creating numpy arrays for each mask.
        
        m = mask
        i = 0
        while m:
            if m & 1:
                current_load += delta_vecs[:, i]
            m >>= 1
            i += 1
            # If m becomes 0, we can break, but we need to ensure i stays in bounds
            if m == 0:
                break
                
        # Note: The while loop above stops when m is 0. 
        # If the highest bits are 0, it stops early. This is correct.
        
        current_max_load = np.max(current_load)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            
    # Construct the result array from best_mask
    best_routing = np.zeros(n_demands, dtype=np.float64)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            best_routing[i] = 1.0
        else:
            best_routing[i] = 0.0
            
    return best_routing
