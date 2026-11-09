import numpy as np
import time

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
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Precompute contributions
    cw_contrib = np.zeros((n_demands, n_nodes))
    ccw_contrib = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s < t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in edges:
            cw_contrib[i, e] = w
            
        # CCW is complement
        # Instead of set difference, we can use a mask
        cw_mask = np.zeros(n_nodes, dtype=bool)
        cw_mask[edges] = True
        ccw_mask = ~cw_mask
        ccw_contrib[i, ccw_mask] = w

    base_load = np.sum(cw_contrib, axis=0)
    deltas = ccw_contrib - cw_contrib

    # Exhaustive search
    masks = np.arange(1 << n_demands, dtype=np.int32)
    # Efficient bit extraction
    bits = (masks[:, None] >> np.arange(n_demands, dtype=np.int32)[None, :]) & 1
    
    loads = base_load + bits @ deltas
    max_loads = loads.max(axis=1)
    best_idx = max_loads.argmin()
    best_mask = masks[best_idx]

    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
