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

    # Build incidence matrices:
    # cw[i, e] = 1 if demand i routed clockwise uses edge e
    # ccw[i, e] = 1 if demand i routed counter-clockwise uses edge e
    cw = np.zeros((D, n), dtype=float)
    ccw = np.zeros((D, n), dtype=float)

    for i, (s, t, w) in enumerate(demands):
        # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
        # Number of edges clockwise: (t - s) mod n
        cw_dist = (t - s) % n
        if cw_dist == 0:
            # This shouldn't happen for distinct nodes, but handle just in case
            pass
        for k in range(cw_dist):
            e = (s + k) % n
            cw[i, e] = 1.0

        # Counter-clockwise uses all other edges
        ccw[i, :] = 1.0 - cw[i, :]

    # Weights
    weights = np.array([w for _, _, w in demands], dtype=float)

    # Decision variables: x_i in {0, 1} for i in range(D)
    # x_i = 0 -> clockwise, x_i = 1 -> counter-clockwise
    # Load on edge e = sum_i weights[i] * (cw[i,e] * (1 - x_i) + ccw[i,e] * x_i)
    #                = sum_i weights[i] * (cw[i,e] + (ccw[i,e] - cw[i,e]) * x_i)
    #                = sum_i weights[i] * cw[i,e] + sum_i weights[i] * (ccw[i,e] - cw[i,e]) * x_i

    # Let A_e = sum_i weights[i] * cw[i,e]  (base load if all clockwise)
    # Let B[i, e] = weights[i] * (ccw[i,e] - cw[i,e])
    # Load_e = A_e + sum_i B[i,e] * x_i

    A = weights @ cw  # shape (n,)
    B = np.zeros((D, n), dtype=float)
    for i in range(D):
        B[i, :] = weights[i] * (ccw[i, :] - cw[i, :])

    # Minimize L subject to:
    # For each edge e: A_e + sum_i B[i,e] * x_i <= L
    # i.e., -L + sum_i B[i,e] * x_i <= -A_e
    # Variables: [L, x_0, x_1, ..., x_{D-1}]
    # c = [1, 0, 0, ..., 0]  (minimize L)
    # Constraints in form A_ub @ var <= b_ub

    # Number of variables: 1 (L) + D (x's)
    n_vars = 1 + D

    c = np.zeros(n_vars)
    c[0] = 1.0  # minimize L

    A_ub = np.zeros((n, n_vars))
    b_ub = np.zeros(n)

    for e in range(n):
        A_ub[e, 0] = -1.0  # -L
        A_ub[e, 1:] = B[:, e]  # sum_i B[i,e] * x_i
        b_ub[e] = -A[e]  # <= -A_e

    # Bounds: L >= 0, x_i in [0, 1]
    bounds = [(0, None)] + [(0, 1)] * D

    # Solve LP relaxation first, then check if integer solution
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        # Fallback: try all-clockwise
        return np.zeros(D)

    x_lp = res.x[1:]

    # Round to nearest integer (should be close to 0 or 1 for this problem)
    x_int = np.round(x_lp)
    x_int = np.clip(x_int, 0, 1)

    # Verify the integer solution is feasible and compute its max load
    def max_load(x):
        loads = A + B @ x
        return np.max(loads)

    max_load_int = max_load(x_int)

    # If the LP solution is already very close to integer, use it
    # Otherwise, try a few local improvements
    best_x = x_int.copy()
    best_load = max_load_int

    # Try flipping individual bits
    for i in range(D):
        trial = best_x.copy()
        trial[i] = 1.0 - trial[i]
        load = max_load(trial)
        if load < best_load:
            best_load = load
            best_x = trial

    # Try pairs of flips
    for i in range(D):
        for j in range(i + 1, D):
            trial = best_x.copy()
            trial[i] = 1.0 - trial[i]
            trial[j] = 1.0 - trial[j]
            load = max_load(trial)
            if load < best_load:
                best_load = load
                best_x = trial

    return best_x


# EVOLVE-BLOCK-END
