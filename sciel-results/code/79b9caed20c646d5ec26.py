import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    n_nodes = 10
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
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute CW and CCW masks
    # cw_mask[i][e] = 1 if demand i uses edge e in CW, else 0
    cw_masks = np.zeros((D, n_nodes), dtype=int)
    ccw_masks = np.zeros((D, n_nodes), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges
        edge = s
        while edge != t:
            cw_masks[i, edge] = 1
            edge = (edge + 1) % n_nodes
        # CCW edges are the complement
        ccw_masks[i] = 1 - cw_masks[i]

    weights = np.array([d[2] for d in demands], dtype=int)
    
    # Base load: all CW
    # load = sum(w_i * cw_mask_i)
    base_load = (cw_masks * weights[:, np.newaxis]).sum(axis=0)
    
    # Delta for switching to CCW
    # If we switch demand i to CCW, the change in load is w_i * (ccw_mask_i - cw_mask_i)
    deltas = (ccw_masks - cw_masks) * weights[:, np.newaxis]
    
    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Exhaustive search
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Calculate load
        # load = base_load + sum(deltas[i] for i where bit i is set)
        # Efficient way:
        # We can iterate bits.
        load = base_load.copy()
        m = mask
        i = 0
        while m:
            if m & 1:
                load += deltas[i]
            m >>= 1
            i += 1
            
        max_load = np.max(load)
        if max_load < best_max_load:
            best_max_load = max_load
            # Extract routing
            best_routing = np.zeros(D)
            m2 = mask
            for j in range(D):
                if m2 & 1:
                    best_routing[j] = 1.0
                m2 >>= 1

    return best_routing
