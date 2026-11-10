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
    deadline = start_time + 600  # 600 second limit, use 590 for safety

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge masks for each demand in both directions
    # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
    # CCW: all other edges

    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_mask = np.zeros(n)
        ccw_mask = np.zeros(n)
        # Clockwise edges
        d = (t - s) % n
        for i in range(d):
            cw_mask[(s + i) % n] = 1
        # Counter-clockwise: all other edges
        ccw_mask = 1 - cw_mask
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)

    cw_masks = np.array(cw_masks)  # shape (D, n)
    ccw_masks = np.array(ccw_masks)  # shape (D, n)
    weights = np.array([w for _, _, w in demands])  # shape (D,)

    # Exhaustive search: 2^15 = 32768 combinations
    best_max_load = float('inf')
    best_routing = np.zeros(D)

    for mask in range(2 ** D):
        # Build routing vector
        routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)

        # Compute edge loads
        # For each demand i, if routing[i] == 0, use cw_masks[i]; else ccw_masks[i]
        # load = sum over i of weights[i] * mask_i
        # Use: for each edge, sum of weights[i] * (routing[i] * ccw_masks[i] + (1-routing[i]) * cw_masks[i])

        # Vectorized: loads = weights @ (routing[:,None] * ccw_masks + (1-routing[:,None]) * cw_masks)
        # Actually: loads[e] = sum_i weights[i] * (routing[i] * ccw_masks[i,e] + (1-routing[i]) * cw_masks[i,e])
        # = sum_i weights[i] * cw_masks[i,e] + sum_i weights[i] * (ccw_masks[i,e] - cw_masks[i,e]) * routing[i]

        base_loads = weights @ cw_masks  # shape (n,)
        delta = weights * (ccw_masks - cw_masks)  # shape (D, n)
        loads = base_loads + delta @ routing  # shape (n,)

        max_load = np.max(loads)

        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()

        # Check time
        if time.time() > deadline - 10:
            break

    return best_routing


# EVOLVE-BLOCK-END
