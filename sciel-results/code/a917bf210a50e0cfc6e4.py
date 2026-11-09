import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep margin

    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute contributions
    # cw_contrib[i, e] is weight of demand i if routed CW, 0 otherwise
    # ccw_contrib[i, e] is weight of demand i if routed CCW, 0 otherwise
    cw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    ccw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_contrib[i, current] = w
            current = (current + 1) % n_nodes
        
        # CCW uses all edges not in the CW path
        ccw_contrib[i, :] = w
        # Zero out the edges that are part of the CW path
        ccw_contrib[i, cw_contrib[i] > 0] = 0
        
    base_loads = np.sum(cw_contrib, axis=0)
    delta_loads = ccw_contrib - cw_contrib
    
    # Vectorized exhaustive search
    n_masks = 1 << n_demands
    masks = np.arange(n_masks, dtype=np.uint16)
    
    # Generate direction matrix (n_masks, n_demands)
    # Bit i of mask determines direction for demand i
    # We want 1 if bit is set (CCW), 0 otherwise (CW)
    directions = ((masks[:, None] >> np.arange(n_demands)) & 1).astype(np.int32)
    
    # Calculate loads for all masks
    # loads = base_loads + directions @ delta_loads
    # directions: (n_masks, n_demands)
    # delta_loads: (n_demands, n_nodes)
    # result: (n_masks, n_nodes)
    loads = base_loads[None, :] + directions @ delta_loads
    
    # Find mask with minimum max load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = directions[best_idx].astype(np.float64)
    
    # Check deadline
    if time.time() > deadline:
        return np.zeros(n_demands, dtype=np.float64)
        
    return best_routing
