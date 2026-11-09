# EVOLVE-BLOCK-START
import time
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n=10 nodes with 10 edges.
    15 demands, each routed CW (0) or CCW (1). Minimize max edge load.

    Returns:
        np.ndarray of shape (15,) with entries 0 (CW) or 1 (CCW) per demand.
    """
    n = 10
    D = 15

    # Demands: (source, target, weight)
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

    # Precompute for each demand d:
    #   cw_edges[d]: which edges (0..9) are used if routed CW
    #   ccw_edges[d]: which edges are used if routed CCW (complement)
    # Represent as a (D, n) binary matrix for CW, and (D, n) for CCW.
    cw_mat = np.zeros((D, n), dtype=np.int32)
    ccw_mat = np.zeros((D, n), dtype=np.int32)

    for d, (s, t, w) in enumerate(demands):
        # CW edges: s, s+1, ..., t-1 (mod n)
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n)) + list(range(0, t))
        for e in cw_edges:
            cw_mat[d, e] = 1
        ccw_mat[d] = 1 - cw_mat[d]

    # Weights vector
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)

    # Enumerate all 2^15 = 32768 routing combinations.
    # For each combination (bitmask), compute edge loads:
    #   load[e] = sum over d: weight[d] * (1 if bit d is 0 else 0) for CW edges
    #            + sum over d: weight[d] * (1 if bit d is 1 else 0) for CCW edges
    #
    # Efficient approach:
    #   Let x[d] = 0 (CW) or 1 (CCW).
    #   load = sum_d weight[d] * ( (1-x[d]) * cw_mat[d] + x[d] * ccw_mat[d] )
    #        = sum_d weight[d] * cw_mat[d] + sum_d weight[d] * x[d] * (ccw_mat[d] - cw_mat[d])
    #
    # Precompute:
    #   base_load[e] = sum_d weight[d] * cw_mat[d, e]   (all CW)
    #   delta[e]     = sum_d weight[d] * x[d] * (ccw_mat[d,e] - cw_mat[d,e])
    #
    # Let diff_mat = ccw_mat - cw_mat  (values in {-1, 0, 1})
    # weighted_diff[d, e] = weight[d] * diff_mat[d, e]
    #
    # For a given x (binary vector of length D):
    #   load = base_load + x @ weighted_diff   (matrix-vector product, shape (n,))

    base_load = cw_mat.T @ weights  # shape (n,)
    diff_mat = ccw_mat - cw_mat     # shape (D, n), values in {-1, 0, 1}
    weighted_diff = diff_mat * weights[:, None]  # shape (D, n)

    # Generate all 2^15 bitmasks as a (32768, 15) binary matrix
    # Use bit manipulation for speed
    num_combos = 1 << D  # 32768
    # Build the binary matrix: for combo i, bit d is (i >> d) & 1
    # Efficient: use np.unpackbits or manual
    # Manual approach with broadcasting
    bits = np.arange(num_combos, dtype=np.uint32)
    # For each demand d, the bit is (bits >> d) & 1
    # Build (num_combos, D) matrix
    x_mat = np.zeros((num_combos, D), dtype=np.float64)
    for d in range(D):
        x_mat[:, d] = ((bits >> d) & 1).astype(np.float64)

    # Compute loads for all combinations:
    # loads[i, e] = base_load[e] + sum_d x_mat[i, d] * weighted_diff[d, e]
    # = base_load + x_mat @ weighted_diff
    loads = x_mat @ weighted_diff + base_load[None, :]  # shape (num_combos, n)

    # Find the combination with minimum max edge load
    max_loads = loads.max(axis=1)  # shape (num_combos,)
    best_idx = int(np.argmin(max_loads))
    best_max_load = max_loads[best_idx]

    # Extract the best routing
    best_bits = bits[best_idx]
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        routing[d] = float((best_bits >> d) & 1)

    return routing


# EVOLVE-BLOCK-END
