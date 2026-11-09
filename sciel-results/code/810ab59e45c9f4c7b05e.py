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
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
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

    # Precompute edge masks for each demand
    # cw_mask[d] is a boolean array of length n: True for edges on clockwise path
    # ccw_mask[d] is True for edges on counter-clockwise path
    cw_masks = np.zeros((D, n), dtype=np.int8)
    ccw_masks = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            # Should not happen, but handle gracefully
            cw_masks[d, :] = 0
            ccw_masks[d, :] = 1
        else:
            # Clockwise: edges s, s+1, ..., t-1 (mod n)
            # If s < t: edges s, s+1, ..., t-1
            # If s > t: edges s, s+1, ..., n-1, 0, ..., t-1
            if s < t:
                cw_mask = np.zeros(n, dtype=np.int8)
                cw_mask[s:t] = 1
            else:
                cw_mask = np.zeros(n, dtype=np.int8)
                cw_mask[s:] = 1
                cw_mask[:t] = 1
            cw_masks[d] = cw_mask
            ccw_masks[d] = 1 - cw_mask

    # Exhaustive search over all 2^15 = 32768 combinations
    # Vectorized approach: iterate over all bitmasks
    best_load = np.inf
    best_routing = np.zeros(D)

    # For speed, precompute contributions
    # For each demand d, if routed clockwise, add weights[d] to cw_masks[d]
    # if routed counter-clockwise, add weights[d] to ccw_masks[d]

    # Total possible combinations: 2^15 = 32768
    # We can vectorize by processing in batches or just loop (32768 is small)

    # Let's do a simple loop with numpy operations for each combination
    # Actually, let's try a smarter vectorized approach

    # Build a matrix where each row is a combination
    # 32768 x 15 matrix of 0s and 1s (0=cw, 1=ccw)
    # Then for each edge, compute load = sum over demands of weight[d] * mask[d][edge] * (direction[d] == mask_value)

    # Let's think: for demand d routed in direction dir[d] (0=cw, 1=ccw),
    # the edge e gets load w[d] if (dir[d]==0 and cw_mask[d][e]==1) or (dir[d]==1 and ccw_mask[d][e]==1)

    # This is: load[e] = sum_d w[d] * (dir[d]*ccw_mask[d][e] + (1-dir[d])*cw_mask[d][e])
    #          = sum_d w[d] * cw_mask[d][e] + sum_d w[d] * dir[d] * (ccw_mask[d][e] - cw_mask[d][e])
    #          = base[e] + sum_d dir[d] * w[d] * diff_mask[d][e]

    # where base[e] = sum_d w[d] * cw_mask[d][e]
    # and diff_mask[d][e] = ccw_mask[d][e] - cw_mask[d][e]

    base = np.zeros(n)
    for d in range(D):
        base += weights[d] * cw_masks[d]

    diff_mask = ccw_masks.astype(np.float64) - cw_masks.astype(np.float64)  # D x n

    # Now for each combination (bitmask of D bits), compute:
    # load = base + dir @ (weights[:,None] * diff_mask)  where dir is 0/1 vector
    # Actually: load[e] = base[e] + sum_d dir[d] * weights[d] * diff_mask[d][e]

    weighted_diff = weights[:, None] * diff_mask  # D x n

    # Generate all 2^15 combinations
    # Use a loop but with vectorized inner computation
    # 32768 iterations, each doing a D x n matrix-vector product (15x10) -> very fast

    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Precompute all direction vectors
    # Using bit manipulation
    for mask in range(1 << D):
        if time.time() > deadline:
            break

        # Extract direction bits
        dir_vec = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)

        # Compute loads: load = base + dir_vec @ weighted_diff
        loads = base + dir_vec @ weighted_diff
        max_load = np.max(loads)

        if max_load < best_load:
            best_load = max_load
            best_routing = dir_vec

    # Return the best routing found
    return best_routing


# EVOLVE-BLOCK-END
