import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing combinations.
    Returns an array of 15 values, where value >= 0.5 means CCW, else CW.
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

    # Pre-compute load vectors for clockwise and counter-clockwise routing
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))

    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_loads[i, current] += w
            current = (current + 1) % n_nodes

        # Counter-clockwise: all other edges
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    # delta_loads[i, e] = ccw_loads[i, e] - cw_loads[i, e]
    # base_load[e] = sum_i cw_loads[i, e]
    # Load for mask m = base_load + sum_{i in bits(m)} delta_loads[i]
    
    delta_loads = ccw_loads - cw_loads
    base_load = cw_loads.sum(axis=0)
    
    # Generate all masks from 0 to 2^15 - 1
    total_masks = 1 << n_demands
    masks = np.arange(total_masks, dtype=np.uint16)
    
    # Compute bits for each mask
    # bits shape: (2^15, 15)
    # bits[m, i] = 1 if bit i of mask m is set
    bits = ((masks[:, None] >> np.arange(n_demands, dtype=np.uint16)) & 1).astype(np.float64)
    
    # Compute loads for all masks
    # loads shape: (2^15, 10)
    loads = base_load[None, :] + bits @ delta_loads
    
    # Find max load per mask
    max_loads = loads.max(axis=1)
    
    # Find the mask with the minimum max load
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Extract routing decisions from best_mask
    # routing[i] = 1.0 if bit i is set (CCW), else 0.0 (CW)
    routing = ((best_mask >> np.arange(n_demands, dtype=np.uint16)) & 1).astype(np.float64)
    
    return routing
