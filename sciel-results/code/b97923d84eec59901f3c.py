import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    """
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
    
    cw_edges = []
    for s, t, w in demands:
        edges = []
        current = s
        while current != t:
            edges.append(current)
            current = (current + 1) % n_nodes
        cw_edges.append(edges)
    
    ccw_edges = []
    for i in range(n_demands):
        cw_set = set(cw_edges[i])
        all_edges = set(range(n_nodes))
        ccw_set = all_edges - cw_set
        ccw_edges.append(list(ccw_set))
    
    weights = np.array([w for s, t, w in demands])
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
            
        loads = np.zeros(n_nodes)
        valid = True
        
        for i in range(n_demands):
            if (mask >> i) & 1:
                edges = ccw_edges[i]
            else:
                edges = cw_edges[i]
            
            w = demands[i][2]
            for e in edges:
                loads[e] += w
                if loads[e] > best_max_load:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            current_max_load = np.max(loads)
            if current_max_load < best_max_load:
                best_max_load = current_max_load
                best_routing = np.array([1.0 if (mask >> i) & 1 else 0.0 for i in range(n_demands)])
            
    return best_routing
