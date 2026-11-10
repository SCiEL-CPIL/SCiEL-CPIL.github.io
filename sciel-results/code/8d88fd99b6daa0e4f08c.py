import numpy as np
import time

def ring_routing() -> np.ndarray:
    t_start = time.time()
    safety_deadline = t_start + 580.0

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

    # Precompute CW and CCW load vectors for each demand
    cw_loads = []
    ccw_loads = []

    for s, t, w in demands:
        cw_edges = []
        if s == t:
            cw_edges = []
        elif s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))

        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]

        cw_vec = np.zeros(n_nodes)
        for e in cw_edges:
            cw_vec[e] = w

        ccw_vec = np.zeros(n_nodes)
        for e in ccw_edges:
            ccw_vec[e] = w

        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)

    cw_arr = np.array(cw_loads)
    ccw_arr = np.array(ccw_loads)

    base = np.sum(cw_arr, axis=0)
    deltas = ccw_arr - cw_arr

    # Vectorized exhaustive search over all 2^15 masks
    num_masks = 1 << n_demands
    masks = np.arange(num_masks)
    bits = ((masks[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    loads = base[None, :] + (bits @ deltas)
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = int(masks[best_idx])
    best_max_load = float(max_loads[best_idx])

    # Local search polish: single-flip and double-flip
    current_bits = np.zeros(n_demands, dtype=np.float64)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            current_bits[i] = 1.0

    current_loads = base + current_bits @ deltas
    current_max = float(np.max(current_loads))

    improved = True
    while improved and time.time() < safety_deadline:
        improved = False
        for i in range(n_demands):
            new_bits = current_bits.copy()
            new_bits[i] = 1.0 - new_bits[i]
            new_loads = base + new_bits @ deltas
            new_max = float(np.max(new_loads))
            if new_max < current_max - 1e-12:
                current_bits = new_bits
                current_loads = new_loads
                current_max = new_max
                improved = True

    improved = True
    while improved and time.time() < safety_deadline:
        improved = False
        for i in range(n_demands):
            if time.time() >= safety_deadline:
                break
            for j in range(i + 1, n_demands):
                if time.time() >= safety_deadline:
                    break
                new_bits = current_bits.copy()
                new_bits[i] = 1.0 - new_bits[i]
                new_bits[j] = 1.0 - new_bits[j]
                new_loads = base + new_bits @ deltas
                new_max = float(np.max(new_loads))
                if new_max < current_max - 1e-12:
                    current_bits = new_bits
                    current_loads = new_loads
                    current_max = new_max
                    improved = True

    if current_max < best_max_load - 1e-12:
        best_max_load = current_max
        best_mask = int(np.sum(current_bits * (1 << np.arange(n_demands))))

    # Construct final result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0

    return routing
