import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Pre-compute CW and CCW load vectors for each demand
    # cw_loads[i, j] is the load added to edge j if demand i is routed CW
    # ccw_loads[i, j] is the load added to edge j if demand i is routed CCW
    cw_loads = np.zeros((D, N))
    ccw_loads = np.zeros((D, N))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s -> s+1 -> ... -> t
        # Edges: s, (s+1)%N, ..., (t-1)%N
        curr = s
        while curr != t:
            cw_loads[i, curr] = w
            curr = (curr + 1) % N
            
        # Counter-clockwise path uses the complement edges
        # Since the ring has N edges and CW uses a subset, CCW uses the rest.
        # Load is w on CCW edges, 0 on CW edges.
        # We can compute this as: w * (1 - cw_mask)
        # But cw_loads[i] has w on CW edges, 0 elsewhere.
        # So ccw_loads[i] = w - cw_loads[i] is correct.
        ccw_loads[i, :] = w - cw_loads[i, :]

    # Optimization:
    # load(mask) = sum(cw_loads[i] if bit_i==0 else ccw_loads[i])
    # Let total_cw = sum(cw_loads)
    # Let diff = ccw_loads - cw_loads
    # load(mask) = total_cw + sum(diff[i] for i where bit_i==1)
    
    total_cw = np.sum(cw_loads, axis=0)
    diff = ccw_loads - cw_loads
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    timeout = 590 # 10s margin
    
    # Iterate through all 2^15 masks
    # Using a simple loop is sufficient for 32768 iterations.
    for mask in range(1 << D):
        # Calculate load vector
        # Start with total_cw
        # Add diff[i] for each set bit i in mask
        load = total_cw.copy()
        
        m = mask
        i = 0
        while m:
            if m & 1:
                load += diff[i]
            m >>= 1
            i += 1
            
        max_load = np.max(load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask
            
        if time.time() - start_time > timeout:
            break

    # Convert best_mask to routing array
    # Entry >= 0.5 routes CCW (bit=1)
    # Entry < 0.5 routes CW (bit=0)
    routing = np.zeros(D)
    for i in range(D):
        if best_mask & (1 << i):
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
