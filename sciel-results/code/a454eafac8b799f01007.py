import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep 10s margin

    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    n_nodes = 10
    n_demands = len(demands)

    # Pre-compute CW and CCW edge contributions for each demand
    # cw_contrib[i, e] is the weight added to edge e if demand i goes CW
    # ccw_contrib[i, e] is the weight added to edge e if demand i goes CCW
    
    cw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_contrib[i, current] = w
            current = (current + 1) % n_nodes
        
    # CCW uses all other edges.
    # Total weight on ring is w. CW uses some edges. CCW uses the rest.
    # Load_CCW(e) = w if e not in CW_path else 0.
    ccw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    for i in range(n_demands):
        w = demands[i][2]
        ccw_contrib[i, :] = w
        ccw_contrib[i, cw_contrib[i] > 0] = 0

    # Base load: all demands routed CW
    base_loads = np.sum(cw_contrib, axis=0)
    
    # Delta load: change in load on edge e if demand i switches from CW to CCW
    # delta[i, e] = ccw_contrib[i, e] - cw_contrib[i, e]
    delta_loads = ccw_contrib - cw_contrib

    # Exhaustive search over 2^15 combinations
    n_masks = 1 << n_demands
    # Generate all masks
    masks = np.arange(n_masks, dtype=np.uint16)
    
    # Determine direction for each demand in each mask
    # directions[m, i] is 1 if demand i is CCW in mask m, 0 if CW
    # Shift right by i, mask with 1
    shifts = np.arange(n_demands)
    directions = ((masks[:, None] >> shifts) & 1).astype(np.int32)

    # Calculate loads for all masks
    # loads[m, e] = base_loads[e] + sum_i directions[m, i] * delta_loads[i, e]
    loads = base_loads[None, :] + (directions @ delta_loads)

    # Find the mask with the minimum maximum edge load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = directions[best_idx].astype(np.float64)

    # Deadline check (should never trigger for this size)
    if time.time() > deadline:
        # Return all CW (zeros) as fallback
        return np.zeros(n_demands, dtype=np.float64)

    return best_routing
