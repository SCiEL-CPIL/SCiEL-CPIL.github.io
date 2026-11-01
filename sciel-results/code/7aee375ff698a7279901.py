# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# Precompute integrals and matrices
_sp = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _sp
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

_M = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 13:
                _M[_k, _i, _j] = _I[_idx]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_bound_and_grad(a0, a4, a8):
    """Compute the uncertainty bound and its gradient with respect to (a0, a4, a8)."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    dp = np.zeros((7, 3))
    dp[0] = [0.0, 0.0, 0.0]
    dp[1] = [8.0, 18.0, 420.0]
    dp[2] = [-14.666667, -43.0, -1330.0]
    dp[3] = [7.333333, 22.0, 742.0]
    dp[4] = [-1.222222, -3.666667, -126.5]
    dp[5] = [66.0/945.0, 198.0/945.0, 6930.0/945.0]
    dp[6] = [-1.0/945.0, -3.0/945.0, -105.0/945.0]

    m = np.einsum('i,kij,j->k', p, _M, p)
    q = p * _SIGNS
    n = np.einsum('i,kij,j->k', q, _M, q) * 2.0 * np.pi

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)

    bound = np.sqrt(vf * vF)

    grad_bound = np.zeros(3)
    for a in range(3):
        dm = np.zeros(3)
        dn = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * np.dot(dp[:, a], np.dot(_M[k], p))
            dn[k] = 2.0 * np.dot(dp[:, a] * _SIGNS, np.dot(_M[k], q)) * 2.0 * np.pi

        dm0, dm2 = dm[0], dm[1]
        dn0, dn2 = dn[0], dn[1]

        dvf = (m0 * dm2 - m2 * dm0) / (m0 * m0)
        dvF = (n0 * dn2 - n2 * dn0) / (n0 * n0)

        grad_bound[a] = 0.5 / bound * (vF * dvf + vf * dvF)

    return bound, grad_bound


def _objective(x):
    b, _ = _compute_bound_and_grad(x[0], x[1], x[2])
    return b


def _objective_grad(x):
    b, g = _compute_bound_and_grad(x[0], x[1], x[2])
    return b, g


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _objective(seed)

    rng = np.random.default_rng(42)

    best_bound = seed_bound
    best_params = seed.copy()

    starts = [seed.copy()]

    # Log-scale random starts
    for _ in range(15):
        a0 = rng.uniform(-5, 5)
        a4 = rng.uniform(-2, 2)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))

    # Structured starts
    for a0 in [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2]:
        for a4 in [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]:
            for a8 in [-0.005, -0.001, 0, 0.001, 0.005]:
                starts.append(np.array([a0, a4, a8]))

    for x0 in starts:
        if time.time() > _DEADLINE - 15:
            break

        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # High-precision polish with Nelder-Mead
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-18})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final verification
    final_bound = _objective(best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
    return best_params
# EVOLVE-BLOCK-END
