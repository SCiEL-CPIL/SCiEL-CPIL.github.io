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

    # Precompute contributions
    cw_contrib = np.zeros((n_demands, n_nodes))
    ccw_contrib = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s < t:
            edges = np.arange(s, t)
        else:
            edges = np.concatenate((np.arange(s, n_nodes), np.arange(0, t)))
        
        cw_contrib[i, edges] = w
        ccw_contrib[i, ~np.isin(np.arange(n_nodes), edges)] = w

    base_load = np.sum(cw_contrib, axis=0)
    deltas = ccw_contrib - cw_contrib

    # Exhaustive search
    masks = np.arange(1 << n_demands, dtype=np.int32)
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
