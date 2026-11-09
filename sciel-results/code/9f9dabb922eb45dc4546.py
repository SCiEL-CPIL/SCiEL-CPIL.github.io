import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute edge sets for each demand
    # cw_edges[i] is a list of edge indices used if demand i is routed clockwise
    # ccw_edges[i] is a list of edge indices used if demand i is routed counter-clockwise
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        # Clockwise path: s -> s+1 -> ... -> t
        cw = []
        curr = s
        while curr != t:
            cw.append(curr)
            curr = (curr + 1) % n_nodes
        
        # Counter-clockwise path: all other edges
        all_edges = set(range(n_nodes))
        cw_set = set(cw)
        ccw = [e for e in all_edges if e not in cw_set]
        
        cw_edges.append(cw)
        ccw_edges.append(ccw)
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search over all 2^15 combinations
    for mask in range(1 << n_demands):
        loads = [0] * n_nodes
        
        for i in range(n_demands):
            if (mask >> i) & 1:
                # Counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
            else:
                # Clockwise
                for e in cw_edges[i]:
                    loads[e] += weights[i]
        
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
            
    # Construct the result array
    # Entry >= 0.5 routes counter-clockwise, else clockwise
    # Our bit 0 means clockwise, bit 1 means counter-clockwise
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
