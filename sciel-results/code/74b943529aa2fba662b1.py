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
    cw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    ccw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    
    for i, (s, t, w) in enumerate(demands):
        if s < t:
            cw_edges = range(s, t)
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        for e in cw_edges:
            cw_loads[i, e] = w
        for e in ccw_edges:
            ccw_loads[i, e] = w
            
    base_load = np.sum(cw_loads, axis=0)
    deltas = ccw_loads - cw_loads
    
    # Exhaustive search over all 2^15 masks
    # This is fast enough: 32768 iterations
    # Compute all loads at once using matrix multiplication
    masks = np.arange(1 << n_demands, dtype=np.uint32)
    bit_positions = np.arange(n_demands, dtype=np.uint32)
    M = (masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1
    
    loads_all = M @ deltas + base_load[np.newaxis, :]
    max_loads = np.max(loads_all, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
            
    return routing
