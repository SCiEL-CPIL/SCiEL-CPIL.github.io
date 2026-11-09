import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    """
    # Demand definitions: (source, target, weight)
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
    
    # Pre-compute edge indicator vectors for clockwise and counter-clockwise routing
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        # For the remaining edges (not on CW path), they are on CCW path
        ccw_vec[i, cw_vec[i] == 0] = 1
    
    # Weights array
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    
    # Precompute the difference matrix: diff[i,e] = ccw_vec[i,e] - cw_vec[i,e]
    # This is 1 if edge e is on CCW but not CW, -1 if on CW but not CCW, 0 otherwise
    diff = ccw_vec.astype(np.int32) - cw_vec.astype(np.int32)
    
    # For each demand i, the contribution to edge loads when routed CW is:
    #   weights[i] * cw_vec[i, :]
    # When routed CCW:
    #   weights[i] * ccw_vec[i, :]
    # 
    # We can precompute the CW base load and the delta per demand
    base_load = (cw_vec.T @ weights)  # shape (n_nodes,)
    delta_loads = (diff.T * weights[:, np.newaxis])  # shape (n_nodes, n_demands)
    # delta_loads[e, i] = weights[i] * diff[i, e]
    # So load[e] = base_load[e] + sum_i delta_loads[e, i] * d_i where d_i is 0 (CW) or 1 (CCW)
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin
    
    # 2^15 = 32768 combinations. This is very fast.
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
            
        # Determine routing direction for each demand
        # bit i of mask: 0 = CW, 1 = CCW
        directions = np.array([(mask >> i) & 1 for i in range(n_demands)], dtype=np.int32)
        
        # load[e] = base_load[e] + sum_i delta_loads[e, i] * directions[i]
        loads = base_load + delta_loads @ directions
        
        current_max_load = int(np.max(loads))
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = directions.astype(np.float64)
    
    return best_routing
