import numpy as np

def ring_routing() -> np.ndarray:
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
    
    # Precompute CW and CCW edge lists for each demand
    cw_edges_list = []
    ccw_edges_list = []
    weights = []
    
    for s, t, w in demands:
        # CW path: s -> s+1 -> ... -> t-1 (mod n_nodes)
        cw_edges = []
        e = s
        while e != t:
            cw_edges.append(e)
            e = (e + 1) % n_nodes
        cw_edges_list.append(cw_edges)
        
        # CCW path: all edges not in CW
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        ccw_edges_list.append(ccw_edges)
        weights.append(w)
    
    # Initial state: all CW
    loads = [0] * n_nodes
    for i in range(n_demands):
        for e in cw_edges_list[i]:
            loads[e] += weights[i]
    
    best_max_load = max(loads)
    best_mask = 0
    
    # Gray code enumeration of all 2^15 = 32768 configurations
    # Consecutive Gray codes differ in exactly one bit, so we only
    # need to update one demand's contribution per step.
    total_masks = 1 << n_demands
    
    for step in range(1, total_masks):
        gray = step ^ (step >> 1)
        prev_gray = (step - 1) ^ ((step - 1) >> 1)
        diff = gray ^ prev_gray
        bit_pos = diff.bit_length() - 1
        
        # Determine new direction for this demand
        new_val = (gray >> bit_pos) & 1
        w = weights[bit_pos]
        
        if new_val == 1:
            # Switching from CW to CCW
            for e in cw_edges_list[bit_pos]:
                loads[e] -= w
            for e in ccw_edges_list[bit_pos]:
                loads[e] += w
        else:
            # Switching from CCW to CW
            for e in ccw_edges_list[bit_pos]:
                loads[e] -= w
            for e in cw_edges_list[bit_pos]:
                loads[e] += w
        
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
    
    # Construct result: >= 0.5 means CCW, < 0.5 means CW
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
    
    return routing
