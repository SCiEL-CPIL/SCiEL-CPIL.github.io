"""Refined Cohn-Goncalves Laguerre double-root construction for the C4 sign-uncertainty bound.

Optimizes the 12 positive double-root locations directly (m=12, n=1) to minimize the
C4 upper bound, which is the ratio of the top Laguerre coefficient to the weighted
sum of all Laguerre coefficients of the monic polynomial with those double roots.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0          # safety margin before the 630 s kill
_DEG = 24               # 2 * m, m = 12


def _build_laguerre_basis():
    """M[k, j] = coeff of x^j in L_k(x); fact[k] = k!."""
    M = np.zeros((_DEG + 1, _DEG + 1))
    for k in range(_DEG + 1):
        for j in range(k + 1):
            M[k, j] = ((-1.0) ** j) * (math.comb(k, j)) / math.factorial(j)
    fact = np.array([float(math.factorial(k)) for k in range(_DEG + 1)])
    return M, fact


_M, _FACT = _build_laguerre_basis()
# Leading coeff of the monic degree-24 polynomial is 1; L_24 has leading coeff 1/24!,
# so a_24 = 24! and the "top" term (a_24 * 24!) is the constant (24!)^2.
_TOP = float(math.factorial(_DEG)) ** 2


def _c4_from_roots(r):
    """Smooth C4 upper bound for the given positive double-root locations."""
    r = np.asarray(r, dtype=float)
    if r.size != 12 or np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        return 1e300
    roots = np.concatenate([r, r])
    try:
        pc = np.poly(roots)            # degree-24 coeffs, highest first
    except Exception:
        return 1e300
    c = pc[::-1].astype(float)         # power basis, ascending
    a = _M @ c                         # Laguerre coefficients a_k
    wsum = float(np.sum(np.abs(a) * _FACT))
    if wsum <= 0.0 or not np.isfinite(wsum):
        return 1e300
    return _TOP / wsum                 # minimize this


def _obj(logr):
    return _c4_from_roots(np.exp(logr))


def construct():
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    best_roots = seed_roots.copy()
    best_val = _c4_from_roots(seed_roots)
    rng = np.random.default_rng(12345)

    def _consider(r, v):
        nonlocal best_roots, best_val
        r = np.sort(np.asarray(r, dtype=float))
        if (r.size == 12 and np.all(np.isfinite(r)) and np.all(r > 0.0)
                and np.all(np.diff(r) > 1e-6) and np.isfinite(v) and v < best_val):
            best_val = v
            best_roots = r

    def _time_ok():
        return time.time() < _DEADLINE - _MARGIN

    base_log = np.log(seed_roots)

    # Multi-restart global search in log-space (positivity guaranteed).
    n_restarts = 60
    for it in range(n_restarts):
        if not _time_ok():
            break
        logr0 = base_log + rng.normal(0.0, 0.7, size=12)
        try:
            res = minimize(_obj, logr0, method='L-BFGS-B',
                           options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        r = np.exp(res.x)
        _consider(r, _c4_from_roots(r))

        # Periodic local refinement around the current best.
        if it % 4 == 0 and _time_ok():
            logr_b = np.log(best_roots) + rng.normal(0.0, 0.05, size=12)
            try:
                res2 = minimize(_obj, logr_b, method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10})
            except Exception:
                continue
            r2 = np.exp(res2.x)
            _consider(r2, _c4_from_roots(r2))

    # Final high-precision polish pass on the best (tighter tol, more iterations).
    if _time_ok():
        logr_p = np.log(best_roots)
        try:
            resp = minimize(_obj, logr_p, method='L-BFGS-B',
                            options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12})
        except Exception:
            resp = None
        if resp is not None:
            rp = np.exp(resp.x)
            _consider(rp, _c4_from_roots(rp))

    return [float(x) for x in best_roots]
