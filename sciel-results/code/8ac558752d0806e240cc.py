# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP (scipy.optimize.milp) with high-precision polish.
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

    # W_cw[d,e] = weight * 1 if demand d uses edge e when routed CW
    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        diff = W_ccw - W_cw
        return W_cw.sum(axis=0) + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 0=CW, 1=CCW), t (continuous, >= 0)
    # Minimize t
    # Subject to: for each edge e:
    #   sum_d (W_cw[d,e]*(1-x_d) + W_ccw[d,e]*x_d) <= t
    #   => sum_d (W_ccw[d,e] - W_cw[d,e])*x_d - t <= -sum_d W_cw[d,e]

    # c: objective coefficients
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t

    # A_ub, b_ub: inequality constraints
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = W_ccw[:, e] - W_cw[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -W_cw[:, e].sum()

    # bounds: x_d in [0,1], t in [0, inf)
    integrality = np.zeros(D + 1)
    integrality[:D] = 1  # x_d are binary
    bounds = [(0, 1)] * D + [(0, None)]

    # Use linprog with highs method (which supports integer variables)
    # Actually, let's use scipy.optimize.milp for clarity
    from scipy.optimize import milp, LinearConstraint, Bounds

    # milp uses: c @ x  (minimize)
    # constraints: LinearConstraint(A, lb, ub)
    # bounds: Bounds(lb, ub)
    # integrality: 1 for integer, 0 for continuous

    cons = LinearConstraint(
        A_ub,
        lb=-np.inf,  # No lower bound on LHS... wait
        ub=b_ub
    )
    # Actually A_ub @ x <= b_ub is equivalent to:
    # A_ub @ x >= -inf, A_ub @ x <= b_ub
    # LinearConstraint(A, lb, ub) means lb <= A @ x <= ub
    # So we want: -inf <= A_ub @ x <= b_ub
    # That's exactly what we have.

    bnds = Bounds(lb=[0]*D + [0], ub=[1]*D + [np.inf])
    integ = np.zeros(D + 1)
    integ[:D] = 1

    try:
        res = milp(
            c=c,
            constraints=cons,
            integrality=integ,
            bounds=bnds
        )
        if res.status == 0:
            best_routing = res.x[:D]
            best_val = res.x[D]
        else:
            # Fallback: use exhaustive search
            best_routing = np.zeros(D, dtype=float)
            best_val = max_load(best_routing)
    except Exception:
        # Fallback: exhaustive search
        best_routing = np.zeros(D, dtype=float)
        best_val = max_load(best_routing)

    # Polish with local search
    improved = True
    while improved:
        improved = False
        cur = max_load(best_routing)
        for d in range(D):
            best_routing[d] = 1.0 - best_routing[d]
            new = max_load(best_routing)
            if new < cur - 1e-12:
                cur = new
                improved = True
            else:
                best_routing[d] = 1.0 - best_routing[d]

    return best_routing

# EVOLVE-BLOCK-END
