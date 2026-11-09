import numpy as np
import time
import math

def ring_routing() -> np.ndarray:
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    D = len(demands)
    
    # Precompute edge lists for CW and CCW for each demand
    cw_edges_list = []
    ccw_edges_list = []
    weights = []
    
    for s, t, w in demands:
        cw_edges = []
        e = s
        while e != t:
            cw_edges.append(e)
            e = (e + 1) % n_nodes
        cw_edges_list.append(cw_edges)
        
        ccw_edges = []
        e = t
        while e != s:
            ccw_edges.append(e)
            e = (e + 1) % n_nodes
        ccw_edges_list.append(ccw_edges)
        
        weights.append(w)
        
    # Initial state: all CW
    loads = np.zeros(n_nodes)
    for i in range(D):
        for e in cw_edges_list[i]:
            loads[e] += weights[i]
            
    best_max_load = np.max(loads)
    best_mask = 0
    current_mask = 0
    current_loads = loads.copy()
    
    # Gray code enumeration
    # Total masks: 2^15 = 32768
    for i in range(1, 1 << D):
        gray = i ^ (i >> 1)
        prev_gray = (i - 1) ^ ((i - 1) >> 1)
        diff = gray ^ prev_gray
        bit_pos = int(math.log2(diff))
        new_val = (gray >> bit_pos) & 1
        
        # Update loads for the flipped demand
        w = weights[bit_pos]
        if new_val == 1:
            # Switching from CW to CCW
            for e in cw_edges_list[bit_pos]:
                current_loads[e] -= w
            for e in ccw_edges_list[bit_pos]:
                current_loads[e] += w
        else:
            # Switching from CCW to CW
            for e in cw_edges_list[bit_pos]:
                current_loads[e] += w
            for e in ccw_edges_list[bit_pos]:
                current_loads[e] -= w
        
        current_max = np.max(current_loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
            # Optional early termination if we hit theoretical lower bound?
            # Lower bound is max(total_weight / n_nodes, max_single_demand_weight)
            # Total weight = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40
            # 40/10 = 4. Max single weight = 5. So LB is 5.
            # We are likely to find something much better than seed (which was ~20).
            # No need for early termination unless we hit 5 or 6.
            # But 32768 iterations is fast.
            
    # Construct result
    result = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
