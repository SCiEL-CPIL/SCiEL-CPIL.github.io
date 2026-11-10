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
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill

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

    # Precompute clockwise and counter-clockwise edge masks for each demand.
    # Edge e is between node e and (e+1) mod n.
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise: the complement.
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(n, dtype=np.int8)
        if s <= t:
            cw[s:t] = 1
        else:
            cw[s:] = 1
            cw[:t] = 1
        ccw = 1 - cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    cw_masks = np.array(cw_masks)  # shape (D, n)
    ccw_masks = np.array(ccw_masks)  # shape (D, n)

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # We will use Gray code enumeration to iterate all 2^D combinations.
    # For each combination, compute edge loads and track the best max load.

    best_max_load = np.inf
    best_mask = np.zeros(D, dtype=np.int8)  # 0 = cw, 1 = ccw

    # We'll use a running load array and update it incrementally via Gray code.
    # In Gray code, only one bit changes per step.

    # Initial state: all clockwise (mask = 0)
    current_mask = np.zeros(D, dtype=np.int8)
    loads = np.zeros(n, dtype=np.float64)
    for i in range(D):
        loads += weights[i] * cw_masks[i]

    current_max_load = np.max(loads)
    if current_max_load < best_max_load:
        best_max_load = current_max_load
        best_mask = current_mask.copy()

    total = 1 << D
    # We've done combination 0 (all zeros). Now iterate 1..total-1.
    for step in range(1, total):
        # Check deadline
        if time.time() > deadline:
            break

        # Determine which bit changed: in Gray code, the bit that changes is the
        # index of the least significant 0 in (step & -step) ... actually:
        # Gray code g = step ^ (step >> 1). The bit that changes from g(step-1) to g(step)
        # is the position of the least significant bit where step has a 1 in its binary
        # representation... Let's compute it properly.
        # The bit that changes is the index of the least significant set bit of step.
        lsb = (step & (-step)).bit_length() - 1

        if current_mask[lsb] == 0:
            # Changing from CW to CCW
            current_mask[lsb] = 1
            loads -= weights[lsb] * cw_masks[lsb]
            loads += weights[lsb] * ccw_masks[lsb]
        else:
            # Changing from CCW to CW
            current_mask[lsb] = 0
            loads -= weights[lsb] * ccw_masks[lsb]
            loads += weights[lsb] * cw_masks[lsb]

        current_max_load = np.max(loads)
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = current_mask.copy()

    # Convert best_mask to the required output format:
    # 0 -> clockwise, 1 -> counter-clockwise (values: <0.5 for CW, >=0.5 for CCW)
    routing = best_mask.astype(np.float64)
    return routing


# EVOLVE-BLOCK-END
