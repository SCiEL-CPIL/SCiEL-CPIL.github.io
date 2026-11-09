import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 600.0 - 10.0  # 590s
    
    n_nodes = 10
    n_demands = 15
    
    # Demands: (source, target, weight)
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
    
    # Precompute contributions
    # cw_contrib[i, e] is weight of demand i on edge e if routed CW
    # ccw_contrib[i, e] is weight of demand i on edge e if routed CCW
    
    cw_contrib = np.zeros((n_demands, n_nodes))
    ccw_contrib = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        # s, s+1, ..., t-1 mod 10
        edges_cw = []
        current = s
        while current != t:
            edges_cw.append(current)
            current = (current + 1) % n_nodes
            
        # Determine CCW edges
        # These are the complement
        edges_ccw = [e for e in range(n_nodes) if e not in edges_cw]
        
        for e in edges_cw:
            cw_contrib[i, e] = w
        for e in edges_ccw:
            ccw_contrib[i, e] = w
            
    # Base load if all routed CW
    base_load = np.sum(cw_contrib, axis=0)
    
    # Delta contribution: change in load on edge e if demand i is switched from CW to CCW
    # Delta[i, e] = ccw_contrib[i, e] - cw_contrib[i, e]
    delta = ccw_contrib - cw_contrib
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    # Iterate over all 2^15 routings
    for mask in range(1 << n_demands):
        # Check time
        if time.time() > deadline:
            break
            
        # Create routing vector x (0 for CW, 1 for CCW)
        # x_i = 1 if bit i of mask is set
        x = np.array([(mask >> i) & 1 for i in range(n_demands)], dtype=float)
        
        # Load = base_load + delta.T @ x
        # delta is (15, 10), x is (15,)
        # delta.T is (10, 15)
        load = base_load + delta.T.dot(x)
        
        current_max_load = np.max(load)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = x.copy()
            
    return best_routing
