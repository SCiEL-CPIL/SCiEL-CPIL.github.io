# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


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

    # Precompute for each demand and each edge whether the edge is used
    # if the demand is routed clockwise (CW) or counter-clockwise (CCW).
    #
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10).
    # Counter-clockwise uses all other edges.
    #
    # We model the decision variable x_d as:
    #   x_d = 0 -> clockwise
    #   x_d = 1 -> counter-clockwise
    #
    # For each edge e, the load is:
    #   sum_d w_d * [edge e is on CW route of d ? (1 - x_d) : x_d]
    # = sum_d w_d * (CW_use[d][e] * (1 - x_d) + CCW_use[d][e] * x_d)
    # = sum_d w_d * CW_use[d][e] + sum_d w_d * (CCW_use[d][e] - CW_use[d][e]) * x_d
    #
    # Note: CCW_use[d][e] = 1 - CW_use[d][e] for each (d, e), since every edge
    # is on exactly one of the two routes. So:
    #   CCW_use[d][e] - CW_use[d][e] = (1 - CW_use[d][e]) - CW_use[d][e] = 1 - 2*CW_use[d][e]
    #
    # Load on edge e = base_load[e] + sum_d w_d * (1 - 2*CW_use[d][e]) * x_d
    #
    # We want to minimize L such that for all e:
    #   base_load[e] + sum_d w_d * (1 - 2*CW_use[d][e]) * x_d <= L
    #   0 <= x_d <= 1 (binary, but LP relaxation may suffice; we'll round or use integer)
    #
    # Since the problem is small, we can solve it as an integer program or just
    # solve the LP and round. Let's first try LP with HiGHS.

    # Build CW_use matrix: D x n
    CW_use = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 mod 10
        if s == t:
            # Full ring? But s != t in our data. Handle just in case.
            CW_use[d, :] = 1
        elif s < t:
            CW_use[d, s:t] = 1
        else:
            CW_use[d, s:] = 1
            CW_use[d, :t] = 1

    # base_load[e] = sum_d w_d * CW_use[d][e]
    weights = np.array([w for _, _, w in demands], dtype=float)
    base_load = CW_use.T @ weights  # shape (n,)

    # Coefficient matrix A: n x D, where A[e, d] = w_d * (1 - 2 * CW_use[d][e])
    A = np.zeros((n, D), dtype=float)
    for d in range(D):
        A[:, d] = weights[d] * (1 - 2 * CW_use[d, :])

    # Minimize L
    # Variables: [L, x_0, x_1, ..., x_14]
    # c = [1, 0, 0, ..., 0]
    c = np.zeros(D + 1)
    c[0] = 1.0

    # Constraints: for each edge e,
    #   -L + sum_d A[e, d] * x_d <= -base_load[e]
    #   => -1 * L + A[e, :] @ x <= -base_load[e]
    #
    # In standard form for linprog (A_ub @ x_var <= b_ub):
    #   x_var = [L, x_0, ..., x_{D-1}]
    #   A_ub row e: [-1, A[e, 0], A[e, 1], ..., A[e, D-1]]
    #   b_ub[e] = -base_load[e]

    A_ub = np.zeros((n, D + 1))
    A_ub[:, 0] = -1.0
    A_ub[:, 1:] = A

    b_ub = -base_load

    # Bounds: L >= 0, 0 <= x_d <= 1
    bounds = [(0, None)] + [(0, 1)] * D

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        # Fallback: all clockwise
        return np.zeros(D)

    x_var = res.x
    x = x_var[1:]  # D values

    # Round to binary
    routing = (x >= 0.5).astype(float)

    return routing


# EVOLVE-BLOCK-END
