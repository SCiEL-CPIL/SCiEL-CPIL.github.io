# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1

    # delta[d, e] = (ccw_mask[d, e] - cw_mask[d, e]) * weights[d]
    # If we flip demand d to CCW, the load on edge e changes by delta[d, e]
    delta = (ccw_mask - cw_mask).astype(np.float64) * weights[:, np.newaxis]

    # Base load when all demands are CW
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0).astype(np.float64)

    # Vectorized exact search over all 2^D combinations
    # D = 15, so 2^15 = 32768 masks
    # We represent all masks as an integer array `masks`
    # We need to evaluate max_load for each mask efficiently.
    
    # Method: 
    # For each edge e, the load is base_load[e] + sum_{d where bit d is 1} delta[d, e]
    # This can be seen as: for each edge e, we sum delta[:, e] for all masks where bit e is set? No, bit d corresponds to demand d.
    # So for a specific mask m, load_e = base_load[e] + sum_{d=0}^{D-1} ((m >> d) & 1) * delta[d, e]
    
    # We can generate the binary matrix M of shape (2^D, D) where M[i, d] = 1 if bit d of i is set.
    # Then loads = base_load[None, :] + M @ delta
    
    # To avoid creating a huge (32768, 15) float matrix if possible, or just do it since it's small:
    # 32768 * 15 * 8 bytes = ~3.9 MB, which is fine.
    
    masks = np.arange(1 << D, dtype=np.uint32)
    
    # Generate M efficiently
    # bits[d] = 1 << d
    bits = (1 << np.arange(D, dtype=np.uint32))
    
    # M[i, d] = (masks[i] & bits[d]) != 0
    # Using broadcasting: masks[:, None] & bits[None, :]
    M = (masks[:, None] & bits[None, :]) != 0
    
    # M is boolean (32768, 15). Convert to float for matrix multiplication
    M_float = M.astype(np.float32)
    
    # Compute loads: (32768, 10)
    # loads[i, e] = base_load[e] + sum_d M_float[i, d] * delta[d, e]
    # This is equivalent to: loads = base_load + M_float @ delta
    
    loads = M_float @ delta
    loads += base_load[None, :]
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_mask_idx = np.argmin(max_loads)
    
    # Extract the best routing from the mask
    # The mask index is best_mask_idx
    # We need to return an array of shape (15,) where 1.0 means CCW (bit set) and 0.0 means CW (bit unset)
    
    best_mask_val = masks[best_mask_idx]
    routing = np.array([1.0 if (best_mask_val >> d) & 1 else 0.0 for d in range(D)])
    
    return routing

# EVOLVE-BLOCK-END
