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
        1 (or >= 0.5) -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n_nodes = 10
    n_edges = 10
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

    # Precompute for each demand:
    #   cw_edges[d]: list of edge indices used if routed clockwise
    #   ccw_edges[d]: list of edge indices used if routed counter-clockwise
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        ccw = []
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        cur = s
        while cur != t:
            cw.append(cur)
            cur = (cur + 1) % n_nodes
        # Counter-clockwise uses all other edges
        all_edges = set(range(n_edges))
        cw_set = set(cw)
        for e in range(n_edges):
            if e not in cw_set:
                ccw.append(e)
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    # Build LP:
    # Variables: x_d for d in 0..D-1 (binary: 0=cw, 1=ccw), and L (max load)
    # Minimize L
    # Subject to: for each edge e, sum over demands d that use edge e of w_d * (indicator) <= L
    #
    # For a demand d routed cw: it uses edges in cw_edges[d]
    # For a demand d routed ccw: it uses edges in ccw_edges[d]
    #
    # Load on edge e = sum_{d: e in cw_edges[d]} w_d * (1 - x_d) + sum_{d: e in ccw_edges[d]} w_d * x_d
    #
    # = sum_{d: e in cw_edges[d]} w_d - sum_{d: e in cw_edges[d]} w_d * x_d + sum_{d: e in ccw_edges[d]} w_d * x_d
    #
    # Let A_e,d be the coefficient of x_d in the load expression for edge e:
    #   if e in ccw_edges[d]: A_e,d = w_d
    #   if e in cw_edges[d]: A_e,d = -w_d
    #   else: A_e,d = 0
    #
    # And b_e = sum_{d: e in cw_edges[d]} w_d
    #
    # Constraint: b_e + sum_d A_e,d * x_d <= L  for all e
    #
    # Rearranged: sum_d A_e,d * x_d - L <= -b_e

    # Number of variables: D + 1 (x_0..x_{D-1}, L)
    n_vars = D + 1

    # Objective: minimize L -> c vector has 0 for x's and 1 for L
    c = np.zeros(n_vars)
    c[D] = 1.0  # L is the last variable

    # Constraints: A_ub @ x <= b_ub
    # For each edge e: sum_d A_e,d * x_d - L <= -b_e
    A_ub = np.zeros((n_edges, n_vars))
    b_ub = np.zeros(n_edges)

    for e in range(n_edges):
        b_e = 0.0
        for d in range(D):
            w = demands[d][2]
            if e in cw_edges[d]:
                A_ub[e, d] = -w
                b_e += w
            elif e in ccw_edges[d]:
                A_ub[e, d] = w
        A_ub[e, D] = -1.0
        b_ub[e] = -b_e

    # Bounds: x_d in [0, 1] for d in 0..D-1, L >= 0
    bounds = [(0, 1)] * D + [(0, None)]

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        x = res.x[:D]
        # Round to binary: x >= 0.5 -> ccw (1), else cw (0)
        routing = (x >= 0.5).astype(float)
    else:
        # Fallback: all clockwise
        routing = np.zeros(D)

    return routing


# EVOLVE-BLOCK-END
