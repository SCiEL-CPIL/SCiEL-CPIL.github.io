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
    
    # Pre-compute edge usage for CW and CCW
    # cw_vec[i, e] = 1 if demand i uses edge e when routed CW
    # ccw_vec[i, e] = 1 if demand i uses edge e when routed CCW
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        ccw_vec[i] = 1 - cw_vec[i]
        
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    # L_cw[e] = sum of weights of demands using edge e in CW
    L_cw = cw_vec.T @ weights
    
    # Delta[i, e] = w_i * (ccw_vec[i, e] - cw_vec[i, e])
    # If we flip demand i from CW to CCW, load on e changes by Delta[i, e]
    Delta = weights[:, np.newaxis] * (ccw_vec - cw_vec)
    
    # Generate all 2^15 masks as binary vectors
    # Shape: (32768, 15)
    masks = np.arange(1 << n_demands)
    D = (masks[:, np.newaxis] >> np.arange(n_demands)) & 1
    
    # Calculate loads for all combinations
    # loads_all[k, e] = L_cw[e] + sum_i D[k, i] * Delta[i, e]
    # loads_all = L_cw[None, :] + D @ Delta
    loads_all = L_cw[None, :] + D @ Delta
    
    # Find the combination with minimum max load
    max_loads = np.max(loads_all, axis=1)
    best_idx = np.argmin(max_loads)
    
    best_routing = D[best_idx].astype(np.float64)
    
    return best_routing
