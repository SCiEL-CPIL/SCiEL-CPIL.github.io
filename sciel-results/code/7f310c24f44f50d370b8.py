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
        0 (or any value < 0.5) -> route the demand clockwise
        1 (or any value >= 0.5) -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute clockwise edge masks for each demand
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            for e in range(s, t):
                cw_mask[d, e] = 1
        else:
            for e in range(s, n):
                cw_mask[d, e] = 1
            for e in range(0, t):
                cw_mask[d, e] = 1
        # Counter-clockwise: all other edges
        ccw_mask[d] = 1 - cw_mask[d]

    total_weight = weights.sum()

    # Exhaustive search over all 2^15 routing combinations
    # For each assignment, compute loads and track best
    best_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    # Use integer arithmetic for speed
    # loads[e] = sum over demands of w_d * mask[d,e]
    # We'll iterate through all 2^15 possibilities using Gray code for efficiency

    # Precompute the contribution vectors
    # For each demand d, cw contribution is w_d * cw_mask[d, :], ccw is w_d * ccw_mask[d, :]
    cw_contrib = weights[:, np.newaxis] * cw_mask.astype(np.float64)  # (D, n)
    ccw_contrib = weights[:, np.newaxis] * ccw_mask.astype(np.float64)  # (D, n)

    # Total load if all clockwise
    base_load = cw_contrib.sum(axis=0)  # (n,)
    # Delta for flipping demand d from CW to CCW
    delta = ccw_contrib - cw_contrib  # (D, n)

    # Gray code enumeration
    # Gray code: g = i ^ (i >> 1); consecutive values differ by one bit
    # The bit that changes going from g to g' is the position of the lowest set bit of (g ^ g')

    best_load_val = float(np.max(base_load))
    best_bits = np.zeros(D, dtype=np.int8)  # 0=cw, 1=ccw

    current_load = base_load.copy()  # start with all CW

    for i in range(1, 1 << D):
        # Find which bit changed
        gray_prev = (i - 1) ^ ((i - 1) >> 1)
        gray_curr = i ^ (i >> 1)
        changed = gray_prev ^ gray_curr
        bit = changed.bit_length() - 1  # index of changed bit

        if gray_curr & changed:
            # Bit turned on: flip from CW to CCW
            current_load += delta[bit]
            best_bits[bit] = 1
        else:
            # Bit turned off: flip from CCW to CW
            current_load -= delta[bit]
            best_bits[bit] = 0

        # Check if this is better
        max_load = float(np.max(current_load))
        if max_load < best_load_val:
            best_load_val = max_load
            best_routing = best_bits.astype(np.float64)

    # Local search polish: try single-bit and double-bit flips from best
    # Start from best found
    current_load = base_load.copy()
    for d in range(D):
        if best_routing[d] >= 0.5:
            current_load += delta[d]

    improved = True
    while improved:
        improved = False
        # Try single-bit flips
        for d in range(D):
            if best_routing[d] < 0.5:
                # Try flipping to CCW
                new_load = current_load + delta[d]
                if np.max(new_load) < best_load_val - 1e-9:
                    current_load = new_load
                    best_routing[d] = 1.0
                    best_load_val = float(np.max(current_load))
                    improved = True
            else:
                # Try flipping to CW
                new_load = current_load - delta[d]
                if np.max(new_load) < best_load_val - 1e-9:
                    current_load = new_load
                    best_routing[d] = 0.0
                    best_load_val = float(np.max(current_load))
                    improved = True

    return best_routing


# EVOLVE-BLOCK-END
