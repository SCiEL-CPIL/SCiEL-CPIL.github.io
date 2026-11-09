# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP formulation with scipy.optimize.milp for provably optimal solution.
    """
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

    # W_cw[d, e] = weight[d] if demand d uses edge e when routed CW, else 0
    # W_ccw[d, e] = weight[d] if demand d uses edge e when routed CCW, else 0
    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]

    start_time = time.time()
    deadline = start_time + 590.0

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        """routing: 0/1 array per demand (0=CW, 1=CCW). Returns edge loads."""
        diff = W_ccw - W_cw
        return W_cw.sum(axis=0) + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 0=CW, 1=CCW), t (continuous, max load)
    # Minimize: t
    # Constraints: for each edge e: sum_d (W_cw[d,e]*(1-x_d) + W_ccw[d,e]*x_d) <= t
    #   => sum_d (W_ccw[d,e]-W_cw[d,e])*x_d - t <= -sum_d W_cw[d,e]

    # c: objective coefficients (minimize t)
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t

    # A_ub: constraint matrix for inequalities A_ub @ vars <= b_ub
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = W_ccw[:, e] - W_cw[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -W_cw[:, e].sum()

    # Bounds: x_d in [0,1] for d=0..D-1, t >= 0
    bounds = Bounds(lb=np.concatenate([np.zeros(D), [0.0]]),
                    ub=np.concatenate([np.ones(D), [np.inf]]))

    # Integrality: x_d are binary (integrality=1), t is continuous (integrality=0)
    integrality = np.concatenate([np.ones(D, dtype=int), [0]])

    # Solve MILP
    best_routing = None
    try:
        res = milp(c=c,
                   constraints=LinearConstraint(A_ub, ub=b_ub),
                   integrality=integrality,
                   bounds=bounds,
                   options={"time_limit": 580.0, "mip_rel_gap": 0.0, "mip_absgap": 1e-12})
        if res.status in [0, 2]:  # 0=optimal, 2=feasible (may be suboptimal)
            best_routing = res.x[:D]
    except Exception:
        pass

    # Fallback: local search if MILP didn't find a solution
    if best_routing is None:
        routing = np.zeros(D, dtype=float)
        improved = True
        while improved:
            improved = False
            cur = max_load(routing)
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
        best_routing = routing.copy()
    else:
        # Round to nearest 0 or 1 for safety
        best_routing = np.where(best_routing >= 0.5, 1.0, 0.0)

    # Local search polish from best found
    if time.time() < deadline - 5.0:
        routing = best_routing.copy()
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            cur = max_load(routing)
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
        if max_load(routing) < max_load(best_routing):
            best_routing = routing.copy()

    return best_routing

# EVOLVE-BLOCK-END
