import time
import numpy as np
from scipy.optimize import minimize, milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using a smooth min-max surrogate
    (log-sum-exp) optimized with L-BFGS-B, then refines with local
    binary search and ILP verification.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

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
    d = len(demands)

    # Pre-compute clockwise and counter-clockwise edge usage matrices
    cw = np.zeros((d, n), dtype=np.float64)
    ccw = np.zeros((d, n), dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)

    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1.0
            e = (e + 1) % n
        ccw[i] = 1.0 - cw[i]

    # Edge load as function of routing x (0=CW, 1=CCW)
    # Load_e = base_e + sum_i w_i * (ccw[i,e] - cw[i,e]) * x_i
    base_load = np.dot(w, cw)  # shape (n,)
    delta = w[:, None] * (ccw - cw)  # shape (d, n)

    def compute_loads(x):
        """Compute edge loads for routing x."""
        return base_load + x @ delta

    def max_load(x):
        """Maximum edge load."""
        return np.max(compute_loads(x))

    def log_sum_exp_loads(x, alpha):
        """Smooth approximation of max load using log-sum-exp."""
        loads = compute_loads(x)
        m = np.max(loads)
        return m + np.log(np.sum(np.exp(alpha * (loads - m)))) / alpha

    def log_sum_exp_grad(x, alpha):
        """Gradient of log-sum-exp approximation."""
        loads = compute_loads(x)
        m = np.max(loads)
        exps = np.exp(alpha * (loads - m))
        softmax = exps / np.sum(exps)
        # d/dx log_sum_exp = delta^T * softmax
        return delta.T @ softmax

    # --- Phase 1: Smooth optimization with L-BFGS-B ---
    best_x = None
    best_score = np.inf

    # Try multiple alpha values and starting points
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    starts = [
        np.zeros(d),
        np.ones(d),
        np.random.default_rng(42).random(d),
        np.random.default_rng(123).random(d),
        np.random.default_rng(7).random(d),
    ]

    for alpha in alphas:
        if time.time() > deadline:
            break
        for start in starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    log_sum_exp_loads,
                    start,
                    args=(alpha,),
                    jac=lambda x, a=alpha: log_sum_exp_grad(x, a),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * d,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.success or res.fun < best_score:
                    # Round to binary
                    x_rounded = (res.x >= 0.5).astype(np.float64)
                    ml = max_load(x_rounded)
                    if ml < best_score:
                        best_score = ml
                        best_x = x_rounded.copy()
            except Exception:
                pass

    # Also try the continuous solution rounded
    if time.time() < deadline:
        try:
            res = minimize(
                log_sum_exp_loads,
                np.full(d, 0.5),
                args=(50.0,),
                jac=lambda x: log_sum_exp_grad(x, 50.0),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * d,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_rounded = (res.x >= 0.5).astype(np.float64)
            ml = max_load(x_rounded)
            if ml < best_score:
                best_score = ml
                best_x = x_rounded.copy()
        except Exception:
            pass

    # --- Phase 2: Local binary search (flip one demand at a time) ---
    if best_x is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(d):
                if time.time() > deadline:
                    break
                x_trial = best_x.copy()
                x_trial[i] = 1.0 - x_trial[i]
                ml = max_load(x_trial)
                if ml < best_score:
                    best_score = ml
                    best_x = x_trial.copy()
                    improved = True
        # Try flipping two demands at a time
        if time.time() < deadline:
            for i in range(d):
                if time.time() > deadline:
                    break
                for j in range(i + 1, d):
                    if time.time() > deadline:
                        break
                    x_trial = best_x.copy()
                    x_trial[i] = 1.0 - x_trial[i]
                    x_trial[j] = 1.0 - x_trial[j]
                    ml = max_load(x_trial)
                    if ml < best_score:
                        best_score = ml
                        best_x = x_trial.copy()
                        improved = True
            # Repeat single flips after double flips
            if improved:
                improved2 = True
                while improved2 and time.time() < deadline:
                    improved2 = False
                    for i in range(d):
                        if time.time() > deadline:
                            break
                        x_trial = best_x.copy()
                        x_trial[i] = 1.0 - x_trial[i]
                        ml = max_load(x_trial)
                        if ml < best_score:
                            best_score = ml
                            best_x = x_trial.copy()
                            improved2 = True

    # --- Phase 3: ILP verification with tighter bounds ---
    if best_x is not None and time.time() < deadline:
        # Try to find a solution with max_load <= best_score - 1 using ILP
        target_L = int(best_score) - 1
        if target_L >= int(np.max(base_load)):
            try:
                A_rows = []
                b_upper = []
                for e in range(n):
                    row = delta[:, e]
                    A_rows.append(row)
                    b_upper.append(target_L - base_load[e])
                A = np.array(A_rows)
                ub = np.array(b_upper)
                lb = np.full(n, -np.inf)
                constraints = LinearConstraint(A, lb, ub)
                bounds = Bounds(np.zeros(d), np.ones(d))
                integrality = np.ones(d, dtype=np.int8)
                res = milp(
                    c=np.zeros(d),
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds
                )
                if res.status == 0 and res.x is not None:
                    x_ilp = (res.x >= 0.5).astype(np.float64)
                    ml = max_load(x_ilp)
                    if ml < best_score:
                        best_score = ml
                        best_x = x_ilp.copy()
            except Exception:
                pass

    # Fallback
    if best_x is None:
        best_x = np.zeros(d, dtype=np.float64)

    return best_x.astype(np.float64)
