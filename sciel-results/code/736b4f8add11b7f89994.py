import numpy as np
import time
from scipy.optimize import linprog


def ring_routing() -> np.ndarray:
    n_nodes = 10
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
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    # Precompute edge incidence: for each demand, which edges are used in CW vs CCW
    # CW from s to t: edges s, s+1, ..., t-1 (mod n)
    # CCW: all other edges
    cw_inc = np.zeros((D, n_nodes), dtype=float)  # 1 if edge used in CW
    ccw_inc = np.zeros((D, n_nodes), dtype=float)  # 1 if edge used in CCW
    weights = np.array([d[2] for d in demands], dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            cw_inc[i, :] = 1.0
        else:
            edge = s
            while edge != t:
                cw_inc[i, edge] = 1.0
                edge = (edge + 1) % n_nodes
        ccw_inc[i, :] = 1.0 - cw_inc[i, :]
    
    # Formulate as LP:
    # Variables: x[0..D-1] in [0,1] (1 = CCW, 0 = CW), t (max load)
    # Minimize: t
    # Subject to: for each edge e: sum_i w_i * (cw_inc[i,e] * (1-x_i) + ccw_inc[i,e] * x_i) <= t
    #            0 <= x_i <= 1
    
    # Rewrite constraint for edge e:
    # sum_i w_i * cw_inc[i,e] * (1 - x_i) + sum_i w_i * ccw_inc[i,e] * x_i <= t
    # sum_i w_i * cw_inc[i,e] - sum_i w_i * cw_inc[i,e] * x_i + sum_i w_i * ccw_inc[i,e] * x_i <= t
    # sum_i w_i * (ccw_inc[i,e] - cw_inc[i,e]) * x_i <= t - sum_i w_i * cw_inc[i,e]
    # -sum_i w_i * (ccw_inc[i,e] - cw_inc[i,e]) * x_i + t >= sum_i w_i * cw_inc[i,e]
    
    # In linprog form (minimize c^T x s.t. A_ub x <= b_ub):
    # Variables: [x_0, ..., x_{D-1}, t]
    # Objective: minimize t -> c = [0, ..., 0, 1]
    # Constraints: for each edge e:
    #   sum_i w_i * (ccw_inc[i,e] - cw_inc[i,e]) * x_i - t <= -sum_i w_i * cw_inc[i,e]
    
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t
    
    A_ub = np.zeros((n_nodes, D + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        # Coefficients for x_i: w_i * (ccw_inc[i,e] - cw_inc[i,e])
        A_ub[e, :D] = weights * (ccw_inc[:, e] - cw_inc[:, e])
        A_ub[e, D] = -1.0  # -t
        b_ub[e] = -np.sum(weights * cw_inc[:, e])
    
    # Bounds: 0 <= x_i <= 1, t >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x[:D]
        # Round to binary: if x_i > 0.5, route CCW (1), else CW (0)
        routing = np.where(x > 0.5, 1.0, 0.0)
    else:
        # Fallback: use a heuristic - route each demand in the direction that minimizes
        # the number of edges used
        routing = np.zeros(D)
        for i, (s, t, w) in enumerate(demands):
            cw_len = 0
            if s != t:
                edge = s
                while edge != t:
                    cw_len += 1
                    edge = (edge + 1) % n_nodes
            ccw_len = n_nodes - cw_len
            if ccw_len < cw_len:
                routing[i] = 1.0
    
    # Verify and ensure valid output
    routing = np.array(routing, dtype=float).flatten()
    assert routing.shape == (D,), f"Wrong shape: {routing.shape}"
    assert np.all(routing >= 0) and np.all(routing <= 1), "Invalid values"
    
    return routing
