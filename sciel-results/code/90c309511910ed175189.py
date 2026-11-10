# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


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
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
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

    # Precompute edge masks for clockwise and counter-clockwise routes
    # clockwise(s, t): edges s, s+1, ..., t-1 (mod n)
    # counter-clockwise(s, t): all other edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw_mask = np.zeros(n)
        if s <= t:
            for e in range(s, t):
                cw_mask[e] = 1
        else:
            for e in range(s, n):
                cw_mask[e] = 1
            for e in range(0, t):
                cw_mask[e] = 1
        ccw_mask = 1 - cw_mask
        cw_edges.append(cw_mask)
        ccw_edges.append(ccw_mask)

    cw_edges = np.array(cw_edges)  # shape (D, n)
    ccw_edges = np.array(ccw_edges)  # shape (D, n)
    weights = np.array([w for _, _, w in demands])  # shape (D,)

    def compute_max_load(routing_bits):
        """
        routing_bits: array of 0/1, 0=clockwise, 1=counter-clockwise
        Returns max edge load.
        """
        # For each demand, if bit=0 use cw_edges, if bit=1 use ccw_edges
        # load on edge e = sum over demands of weight * (edge is on route)
        # We can compute: for each demand d, route_mask = (1-bit)*cw_edges[d] + bit*ccw_edges[d]
        # load = sum_d weight[d] * route_mask[d]
        # Using broadcasting:
        # bits shape (D,), cw_edges shape (D, n), ccw_edges shape (D, n)
        # route_mask = (1 - bits[:, None]) * cw_edges + bits[:, None] * ccw_edges  shape (D, n)
        # load = route_mask.T @ weights  shape (n,)
        bits = routing_bits
        route_mask = (1 - bits[:, None]) * cw_edges + bits[:, None] * ccw_edges
        load = route_mask.T @ weights
        return np.max(load)

    # Exhaustive search over all 2^15 = 32768 combinations
    best_max_load = np.inf
    best_routing = np.zeros(D)

    deadline = time.time() + 600  # safety margin
    start_time = time.time()

    for bits_int in range(1 << D):
        if time.time() > deadline:
            break
        bits = np.array([(bits_int >> d) & 1 for d in range(D)], dtype=np.float64)
        ml = compute_max_load(bits)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = bits.copy()

    return best_routing


# EVOLVE-BLOCK-END
