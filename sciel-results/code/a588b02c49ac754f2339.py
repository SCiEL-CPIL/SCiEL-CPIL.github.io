# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses Gray-code enumeration over all 2^15 routing combinations with incremental
    load updates to find the optimal solution minimizing the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin for 630s limit

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

    # Precompute the edge sets for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n_nodes
        cw_edges.append(cw)
        # Counter-clockwise: all other edges
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        ccw_edges.append(ccw)

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Compute base loads (all clockwise)
    base_loads = np.zeros(n_nodes, dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            base_loads[e] += weights[i]

    # Compute delta loads: change in edge loads when demand i switches from CW to CCW
    # delta[i, e] = weight[i] if e in ccw_edges[i] else -weight[i] if e in cw_edges[i] else 0
    delta_loads = np.zeros((D, n_nodes), dtype=np.float64)
    for i in range(D):
        for e in ccw_edges[i]:
            delta_loads[i, e] = weights[i]
        for e in cw_edges[i]:
            delta_loads[i, e] = -weights[i]

    n_masks = 1 << D
    best_max_load = float('inf')
    best_mask = 0

    # Gray-code enumeration
    # Gray code: g = i ^ (i >> 1)
    # Consecutive Gray codes differ in exactly one bit
    current_mask = 0
    current_loads = base_loads.copy()

    for i in range(n_masks):
        if i > 0:
            # Find the bit that changes between gray(i-1) and gray(i)
            # gray(i) = i ^ (i >> 1)
            # The bit that flips is the position of the least significant 1 in i
            gray_prev = (i - 1) ^ ((i - 1) >> 1)
            gray_curr = i ^ (i >> 1)
            diff = gray_prev ^ gray_curr
            bit_pos = (diff & -diff).bit_length() - 1  # Position of lowest set bit

            # Update loads: if bit went from 0 to 1, add delta; if 1 to 0, subtract delta
            if (gray_curr >> bit_pos) & 1:
                current_loads += delta_loads[bit_pos]
            else:
                current_loads -= delta_loads[bit_pos]

        current_max_load = np.max(current_loads)

        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = gray_curr

        # Check deadline every 1024 iterations
        if i % 1024 == 0 and time.time() > deadline:
            break

    # Reconstruct best routing from best_mask
    best_routing = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_routing


# EVOLVE-BLOCK-END
