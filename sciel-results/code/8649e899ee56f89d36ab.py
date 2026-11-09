# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP (scipy.optimize.milp) with high-precision polish.
    """
    t_start = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge usage matrices
    A_cw = np.zeros((D, n), dtype=float)
    A_ccw = np.zeros((D, n), dtype=float)
    weights = np.zeros(D, dtype=float)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        e = s
        while e != t:
            A_cw[d, e % n] = 1
            e = (e + 1) % n
        A_ccw[d] = 1.0 - A_cw[d]

    # W_cw[d,e] = weight * 1 if demand d uses edge e when routed CW
    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]

    # Precompute base load (all CW) and diff
    base_load = W_cw.sum(axis=0)
    diff = W_ccw - W_cw  # (D, n)

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        return base_load + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 0=CW, 1=CCW), t (continuous, >= 0)
    # Minimize t
    # Subject to: for each edge e:
    #   sum_d (W_cw[d,e]*(1-x_d) + W_ccw[d,e]*x_d) <= t
    #   => sum_d (W_ccw[d,e] - W_cw[d,e])*x_d - t <= -sum_d W_cw[d,e]

    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t

    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = W_ccw[:, e] - W_cw[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -W_cw[:, e].sum()

    cons = LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    bnds = Bounds(lb=[0]*D + [0], ub=[1]*D + [np.inf])
    integ = np.zeros(D + 1)
    integ[:D] = 1

    best_routing = None
    best_val = np.inf

    try:
        res = milp(
            c=c,
            constraints=cons,
            integrality=integ,
            bounds=bnds
        )
        if res.status == 0:
            best_routing = res.x[:D].copy()
            best_val = max_load(best_routing)
        else:
            best_routing = np.zeros(D, dtype=float)
            best_val = max_load(best_routing)
    except Exception:
        best_routing = np.zeros(D, dtype=float)
        best_val = max_load(best_routing)

    # Polish with local search: 1-flip and 2-flip moves
    def local_search(routing):
        cur = max_load(routing)
        improved = True
        while improved and (time.time() - t_start) < DEADLINE:
            improved = False
            # 1-flip moves
            for d in range(D):
                if (time.time() - t_start) > DEADLINE:
                    break
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
            # 2-flip moves
            for d1 in range(D):
                if (time.time() - t_start) > DEADLINE:
                    break
                for d2 in range(d1 + 1, D):
                    if (time.time() - t_start) > DEADLINE:
                        break
                    routing[d1] = 1.0 - routing[d1]
                    routing[d2] = 1.0 - routing[d2]
                    new = max_load(routing)
                    if new < cur - 1e-12:
                        cur = new
                        improved = True
                    else:
                        routing[d1] = 1.0 - routing[d1]
                        routing[d2] = 1.0 - routing[d2]
        return routing, cur

    if (time.time() - t_start) < DEADLINE:
        best_routing, best_val = local_search(best_routing)

    return best_routing

# EVOLVE-BLOCK-END
