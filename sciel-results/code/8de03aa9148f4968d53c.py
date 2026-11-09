import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using vectorized exhaustive search over all 2^15 routing combinations.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Demand definitions: (source, target, weight)
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

    # Precompute for vectorized exhaustive search
    # For mask m, loads[e] = sum_i (routing_i ? ccw_loads[i,e] : cw_loads[i,e])
    # = cw_total[e] + sum_i (m_i * (ccw_loads[i,e] - cw_loads[i,e]))
    cw_total = cw_loads.sum(axis=0)  # shape (10,)
    delta = ccw_loads - cw_loads     # shape (15, 10)

    # Vectorized exhaustive search over all 2^15 masks
    n_masks = 1 << n_demands  # 32768
    batch_size = 8192
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    shift_arange = np.arange(n_demands, dtype=np.int32)

    for batch_start in range(0, n_masks, batch_size):
        if time.time() > deadline:
            break

        batch_end = min(batch_start + batch_size, n_masks)
        masks = np.arange(batch_start, batch_end, dtype=np.int32)

        # Extract bits: bit i of mask corresponds to demand i
        # bits shape: (batch_size, n_demands), values 0 or 1
        bits = ((masks[:, None] >> shift_arange[None, :]) & 1).astype(np.float64)

        # loads = cw_total[None, :] + bits @ delta.T
        # delta.T shape: (10, 15), bits shape: (batch_size, 15)
        # result shape: (batch_size, 10)
        loads = cw_total[None, :] + bits @ delta.T

        max_loads = loads.max(axis=1)  # shape (batch_size,)

        # Find the best in this batch
        min_idx = np.argmin(max_loads)
        min_val = max_loads[min_idx]

        if min_val < best_max_load:
            best_max_load = min_val
            mask_val = masks[min_idx]
            best_routing = np.array([1.0 if (mask_val >> i) & 1 else 0.0 for i in range(n_demands)])
            if best_max_load <= 4:
                return best_routing

    return best_routing
