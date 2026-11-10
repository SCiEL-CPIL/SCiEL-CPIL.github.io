# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _keich_init(N):
    """Binary Keich construction for 1-D positions."""
    k = int(np.log2(N))
    vals = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = np.array([int(d) for d in bin(idx)[2:].zfill(k)])
        weights = np.array([(1 - j) * (2.0 ** -j) for j in range(1, k + 1)])
        vals[idx] = np.dot(bits, weights) / k
    return vals


def _exact_1d_union_integral(x, N, n_c=2000):
    """
    Compute the exact (via fine grid) integral of the 1-D union length over c in [0,1].
    For each c, intervals are [x[i] + c*i/N, x[i] + c*i/N + 1/N].
    We return the average union length over c (i.e., the integral since c ranges over [0,1]).
    """
    w = 1.0 / N
    idx = np.arange(N, dtype=np.float64)
    c_vals = np.linspace(0.0, 1.0, n_c)
    total = 0.0
    for c in c_vals:
        starts = x + c * idx / N
        order = np.argsort(starts)
        s = starts[order]
        e = s + w
        # Merge intervals
        gaps = np.maximum(0.0, s[1:] - e[:-1])
        union_len = (e[-1] - s[0]) - np.sum(gaps)
        total += union_len
    return total / n_c


def _exact_1d_union_integral_grad(x, N, n_c=2000):
    """
    Compute the integral and its gradient using finite differences or analytical approach.
    For efficiency, we use a central difference for gradient.
    """
    val = _exact_1d_union_integral(x, N, n_c)
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_exact_1d_union_integral(xp, N, n_c) - _exact_1d_union_integral(xm, N, n_c)) / (2 * eps)
    return val, grad


def _optimize_1d(N, x0, budget):
    """Optimize 1-D positions using differential evolution + L-BFGS-B polish."""
    t0 = time.time()
    deadline = t0 + budget

    best_x = x0.copy()
    best_val = _exact_1d_union_integral(best_x, N, n_c=1000)

    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 10:
        try:
            bounds = [(-1.0, 2.0)] * N

            def obj(x):
                return _exact_1d_union_integral(x, N, n_c=500)

            res = differential_evolution(
                obj, bounds, seed=12345, maxiter=150, tol=1e-8,
                popsize=12, init='sobol', polish=False, updating='immediate'
            )
            if res.fun < best_val:
                best_x = res.x.copy()
                best_val = obj(best_x)
        except Exception:
            pass

    # Phase 2: L-BFGS-B with numerical gradient for local polish
    if time.time() < deadline - 5:
        try:
            def obj_lbfgs(x):
                return _exact_1d_union_integral(x, N, n_c=2000)

            res2 = minimize(
                obj_lbfgs, best_x, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_val:
                best_x = res2.x.copy()
                best_val = res2.fun
        except Exception:
            pass

    # Phase 3: Nelder-Mead with tighter tolerance for final polish
    if time.time() < deadline - 2:
        try:
            def obj_nm(x):
                return _exact_1d_union_integral(x, N, n_c=3000)

            res3 = minimize(
                obj_nm, best_x, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res3.fun < best_val:
                best_x = res3.x.copy()
                best_val = res3.fun
        except Exception:
            pass

    return best_x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    x0 = _keich_init(N)

    total_budget = 600.0
    safety = 30.0
    per_dim = (total_budget - safety) / 2.0

    t0 = time.time()
    x_opt = _optimize_1d(N, x0, per_dim)

    # Use the same optimized positions for y (symmetric problem)
    y_opt = x_opt.copy()

    i_idx = np.repeat(np.arange(N, dtype=np.int64), N)
    j_idx = np.tile(np.arange(N, dtype=np.int64), N)

    base_x = x_opt[i_idx].astype(np.float64)
    base_y = y_opt[j_idx].astype(np.float64)

    positions = np.column_stack([base_x, base_y])

    return positions


# EVOLVE-BLOCK-END
