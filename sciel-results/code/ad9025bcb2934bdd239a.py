# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses MILP + smooth surrogate + multi-pass local search with deadline guard.
    """
    start_time = time.time()
    DEADLINE = 600.0  # seconds

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
    base_loads = W_cw.sum(axis=0)
    diff = W_ccw - W_cw  # diff[d,e] = W_ccw[d,e] - W_cw[d,e]

    def compute_loads(routing: np.ndarray) -> np.ndarray:
        return base_loads + diff.T @ routing

    def max_load(routing: np.ndarray) -> float:
        return float(np.max(compute_loads(routing)))

    def logsumexp_smooth(routing: np.ndarray, beta: float = 10.0) -> float:
        """Smooth surrogate for max load using log-sum-exp."""
        loads = compute_loads(routing)
        m = np.max(loads)
        return m + np.log(np.sum(np.exp(beta * (loads - m)))) / beta

    # MILP formulation
    c = np.zeros(D + 1)
    c[D] = 1.0

    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = diff[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_loads[e]

    cons = LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    bnds = Bounds(lb=[0]*D + [0], ub=[1]*D + [np.inf])
    integ = np.zeros(D + 1)
    integ[:D] = 1

    best_routing = None
    best_val = float('inf')

    # Phase 1: MILP
    try:
        res = milp(c=c, constraints=cons, integrality=integ, bounds=bnds)
        if res.status == 0:
            best_routing = res.x[:D].copy()
            best_val = max_load(best_routing)
    except Exception:
        pass

    if best_routing is None:
        best_routing = np.zeros(D, dtype=float)
        best_val = max_load(best_routing)

    # Phase 2: Smooth surrogate optimization to find alternative starting points
    if time.time() - start_time < DEADLINE - 60:
        for beta in [5.0, 10.0, 20.0, 50.0]:
            if time.time() - start_time >= DEADLINE - 60:
                break
            # Gradient descent on the smooth surrogate
            x = best_routing.copy()
            lr = 0.1
            for it in range(200):
                if time.time() - start_time >= DEADLINE - 60:
                    break
                # Numerical gradient
                grad = np.zeros(D)
                for d in range(D):
                    x[d] += 1e-4
                    f_plus = logsumexp_smooth(x, beta)
                    x[d] -= 2e-4
                    f_minus = logsumexp_smooth(x, beta)
                    x[d] += 1e-4
                    grad[d] = (f_plus - f_minus) / (2e-4)
                x -= lr * grad
                # Clip to [0,1]
                x = np.clip(x, 0, 1)
                # Binary rounding
                x_bin = (x > 0.5).astype(float)
                val = max_load(x_bin)
                if val < best_val - 1e-12:
                    best_val = val
                    best_routing = x_bin.copy()

    # Phase 3: Multi-pass local search (1-flip then 2-flip)
    def local_search(routing: np.ndarray) -> np.ndarray:
        """Perform 1-flip and 2-flip local search."""
        routing = routing.copy()
        cur = max_load(routing)
        improved = True
        while improved:
            improved = False
            # 1-flip
            for d in range(D):
                routing[d] = 1.0 - routing[d]
                new = max_load(routing)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                else:
                    routing[d] = 1.0 - routing[d]
            # 2-flip
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    routing[d1] = 1.0 - routing[d1]
                    routing[d2] = 1.0 - routing[d2]
                    new = max_load(routing)
                    if new < cur - 1e-12:
                        cur = new
                        improved = True
                    else:
                        routing[d1] = 1.0 - routing[d1]
                        routing[d2] = 1.0 - routing[d2]
        return routing

    best_routing = local_search(best_routing)
    best_val = max_load(best_routing)

    # Phase 4: Try multiple random restarts with local search
    rng = np.random.default_rng(42)
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() - start_time >= DEADLINE - 30:
            break
        x_rand = (rng.random(D) > 0.5).astype(float)
        x_ls = local_search(x_rand)
        val = max_load(x_ls)
        if val < best_val - 1e-12:
            best_val = val
            best_routing = x_ls.copy()

    # Phase 5: Final exhaustive neighborhood check (all single flips from best)
    final = best_routing.copy()
    for d in range(D):
        final[d] = 1.0 - final[d]
        if max_load(final) < max_load(best_routing) - 1e-12:
            best_routing = final.copy()
        final[d] = 1.0 - final[d]

    return best_routing

# EVOLVE-BLOCK-END
