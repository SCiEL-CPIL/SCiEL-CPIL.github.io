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
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    DEADLINE = start_time + 590.0  # 40s safety margin

    n_nodes = 10
    n_edges = 10
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

    # Precompute clockwise edge masks for each demand
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n_edges, dtype=np.int8)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, n_nodes):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)

    ccw_masks = [1 - m for m in cw_masks]
    weights = np.array([w for _, _, w in demands], dtype=np.int64)

    # base_load: load if all demands go clockwise
    base_load = np.zeros(n_edges, dtype=np.int64)
    # deltas[i]: change in load vector when demand i switches from cw to ccw
    deltas = np.zeros((D, n_edges), dtype=np.int64)
    for i in range(D):
        base_load += weights[i] * cw_masks[i]
        deltas[i] = weights[i] * (ccw_masks[i] - cw_masks[i])

    # Exhaustive search over all 2^15 = 32768 combinations
    # This is fast enough to complete well within deadline
    best_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    for mask_int in range(1 << D):
        if time.time() > DEADLINE:
            break

        # Compute load vector: base_load + sum of deltas for ccw demands
        x = np.array([(mask_int >> i) & 1 for i in range(D)], dtype=np.int64)
        load = base_load + x @ deltas
        max_load = np.max(load)

        if max_load < best_load:
            best_load = max_load
            best_routing = x.astype(np.float64)

    # Return as 0/1 array (entries >= 0.5 mean ccw, < 0.5 means cw)
    return best_routing

# EVOLVE-BLOCK-END
