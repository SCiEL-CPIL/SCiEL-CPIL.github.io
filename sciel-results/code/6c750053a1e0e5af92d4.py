# EVOLVE-BLOCK-START
import numpy as np


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
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute clockwise edge sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        # Counter-clockwise: all other edges
        all_edges = set(range(n))
        ccw = all_edges - cw
        ccw_edges.append(ccw)

    # Brute force all 2^15 combinations
    best_load = float('inf')
    best_routing = np.zeros(D)

    for mask in range(1 << D):
        loads = np.zeros(n)
        for i in range(D):
            if (mask >> i) & 1:
                # Counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += demands[i][2]
            else:
                # Clockwise
                for e in cw_edges[i]:
                    loads[e] += demands[i][2]
        max_load = np.max(loads)
        if max_load < best_load:
            best_load = max_load
            best_routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)

    return best_routing


# EVOLVE-BLOCK-END
