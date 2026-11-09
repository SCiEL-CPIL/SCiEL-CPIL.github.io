import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by minimizing the maximum edge load.
    Uses a vectorized exhaustive search over all 2^15 routing combinations.
    """
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute load contributions
    # load_contrib[i, 0] is CW contribution, load_contrib[i, 1] is CCW contribution
    load_contrib = np.zeros((n_demands, 2, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        cw_edges = []
        current = s
        while current != t:
            cw_edges.append(current)
            current = (current + 1) % n_nodes
        
        load_contrib[i, 0, cw_edges] = w
        
        all_edges = np.arange(n_nodes)
        ccw_edges = np.array([e for e in all_edges if e not in cw_edges])
        load_contrib[i, 1, ccw_edges] = w

    base_load = np.sum(load_contrib[:, 0, :], axis=0)
    delta = load_contrib[:, 1, :] - load_contrib[:, 0, :] # shape (15, 10)
    
    # Vectorized evaluation of all masks
    # Generate all masks from 0 to 2^15 - 1
    masks = np.arange(1 << n_demands, dtype=np.uint32)
    
    # Create a binary matrix where bit i of mask m is in masks_bits[m, i]
    # We want masks_bits[m, i] = (m >> i) & 1
    bits = (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
    
    # Calculate loads for all masks
    # loads[m, j] = base_load[j] + sum(bits[m, i] * delta[i, j])
    loads = base_load[None, :] + (bits @ delta)
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result array
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
