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
    
    # Pre-compute base_load (all CW) and delta (CCW - CW) for each demand
    base_load = np.zeros(n_nodes)
    delta = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges: s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            base_load[current] += w
            # CCW uses all other edges, so delta = CCW - CW
            # If edge is on CW path: CCW doesn't use it, CW does -> delta = -w
            # If edge is NOT on CW path: CCW uses it, CW doesn't -> delta = +w
            # We'll handle this after marking CW edges
            current = (current + 1) % n_nodes
        
        # For delta: edges on CW path get -w, edges NOT on CW path get +w
        # First set all to +w (CCW contribution)
        delta[i, :] = w
        # Then set CW edges to -w (subtract CW contribution)
        current = s
        while current != t:
            delta[i, current] = -w
            current = (current + 1) % n_nodes
    
    # Brute force over all 2^15 masks
    # For mask, loads = base_load + sum(delta[i] for i where bit i is set)
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Pre-compute bit positions for faster lookup
    # Use a simple but fast approach: iterate masks and accumulate
    
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
        
        # Compute loads efficiently using bit operations
        loads = base_load.copy()
        m = mask
        i = 0
        while m:
            if m & 1:
                loads += delta[i]
            m >>= 1
            i += 1
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
    
    # Build result: 1.0 for CCW (>= 0.5), 0.0 for CW
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
