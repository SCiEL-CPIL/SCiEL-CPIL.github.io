import numpy as np
import time

def ring_routing() -> np.ndarray:
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
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute load contributions for each demand in CW and CCW
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        if s < t:
            cw_edges = list(range(s, t))
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_edges = list(all_edges - cw_set)
        
        cw_vec = [0] * n_nodes
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = [0] * n_nodes
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
        
    best_max_load = float('inf')
    best_mask = 0
    
    # Use Gray code traversal for incremental updates
    # Gray code: g(i) = i ^ (i >> 1)
    # Consecutive Gray codes differ in exactly one bit
    
    # Start with mask = 0 (all CW)
    loads = [0] * n_nodes
    for i in range(n_demands):
        for e in range(n_nodes):
            loads[e] += cw_loads[i][e]
    
    current_max = max(loads)
    best_max_load = current_max
    best_mask = 0
    
    start_time = time.time()
    timeout = 600.0
    safety_margin = 30.0
    
    total_masks = 1 << n_demands
    
    # Track which direction each demand is currently using
    # 0 = CW, 1 = CCW
    directions = [0] * n_demands
    
    for step in range(1, total_masks):
        # Check time
        if (time.time() - start_time) > (timeout - safety_margin):
            break
            
        # Gray code for this step
        gray = step ^ (step >> 1)
        
        # Find which bit changed between previous gray and current gray
        prev_gray = (step - 1) ^ ((step - 1) >> 1)
        diff = gray ^ prev_gray
        # diff has exactly one bit set
        bit_index = diff.bit_length() - 1
        
        # Update loads incrementally
        if directions[bit_index] == 0:
            # Was CW, now CCW: remove cw_loads, add ccw_loads
            for e in range(n_nodes):
                loads[e] -= cw_loads[bit_index][e]
                loads[e] += ccw_loads[bit_index][e]
            directions[bit_index] = 1
        else:
            # Was CCW, now CW: remove ccw_loads, add cw_loads
            for e in range(n_nodes):
                loads[e] -= ccw_loads[bit_index][e]
                loads[e] += cw_loads[bit_index][e]
            directions[bit_index] = 0
        
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
            
    # Construct result array
    # Entry >= 0.5 means CCW (1), < 0.5 means CW (0)
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
