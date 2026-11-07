"""Uncertainty inequality: minimize r^2/(2*pi) via two-stage global+local search.

Structural change from the fixed seed: a randomized global scan over the
(c0,c1,c2) coefficients followed by a Nelder-Mead local polish and a
high-precision Newton refinement of the relevant quintic root.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton

_PI = np.pi
_T0 = time.time()
_DEADLINE = 200.0          # wall-clock safety margin (evaluator kills at 210 s)
_EPS = 1e-11


def _q_coeffs(c0, c1, c2):
    """Return [a5,a4,a3,a2,a1,a0] for q(x)=g/x^2 (quintic)."""
    c3 = (8.0 * c0 - 15.0 * c1 + 315.0 * c2) / 792.0
    return np.array([
        144.0 * c3,
        0.0,
        -4320.0 * c2,
        180.0 * c1,
        -8.0 * c0,
        48.0 * c1 - 10800.0 * c2,
    ])


def _value(c0, c1, c2):
    """value = r^2/(2*pi); r = largest sign-changing real root of q. Inf if none."""
    a = _q_coeffs(c0, c1, c2)
    if abs(a[0]) < 1e-15:
        return np.inf
    try:
        roots = np.roots(a)
    except Exception:
        return np.inf
    real = roots[np.abs(roots.imag) < 1e-8].real
    if real.size == 0:
        return np.inf
    best = None
    for r in real:
        if abs(r) < _EPS:
            continue
        # sign change test: q(r-eps)*q(r+eps) < 0
        qm = np.polyval(a, r - 1e-7)
        qp = np.polyval(a, r + 1e-7)
        if qm * qp < 0.0:
            if best is None or r > best:
                best = r
    if best is None:
        return np.inf
    return (best * best) / (2.0 * _PI)


def _polish(c0, c1, c2):
    """High-precision Newton refinement of the largest sign-changing root."""
    a = _q_coeffs(c0, c1, c2)
    if abs(a[0]) < 1e-15:
        return _value(c0, c1, c2)
    try:
        roots = np.roots(a)
    except Exception:
        return _value(c0, c1, c2)
    real = roots[np.abs(roots.imag) < 1e-8].real
    if real.size == 0:
        return np.inf
    cands = []
    for r in real:
        if abs(r) < _EPS:
            continue
        if np.polyval(a, r - 1e-7) * np.polyval(a, r + 1e-7) < 0.0:
            cands.append(r)
    if not cands:
        return np.inf
    r = max(cands)
    try:
        r = newton(lambda x: np.polyval(a, x), r, maxiter=60, tol=1e-14)
    except Exception:
        pass
    return (r * r) / (2.0 * _PI)


def construct():
    deadline = _T0 + _DEADLINE
    rng = np.random.default_rng(12345)

    # --- Stage 1: randomized global scan -----------------------------------
    best = _value(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    n_global = 60000
    for i in range(n_global):
        if time.time() > deadline - 40.0:
            break
        # mix of a wide uniform box and a tight box around the seed
        if (i % 5) == 0:
            c0 = -3.0 + 6.0 * rng.random()
            c1 = -1.0 + 2.0 * rng.random()
            c2 = -0.2 + 0.4 * rng.random()
        else:
            c0 = -1.13 + 0.9 * (2.0 * rng.random() - 1.0)
            c1 = 0.04 + 0.12 * (2.0 * rng.random() - 1.0)
            c2 = 0.0003086 + 0.0005 * (2.0 * rng.random() - 1.0)
        v = _value(c0, c1, c2)
        if v < best:
            best = v
            best_c = np.array([c0, c1, c2])

    # --- Stage 2: Nelder-Mead local polish (multiple restarts) --------------
    def obj(x):
        v = _value(x[0], x[1], x[2])
        return v if np.isfinite(v) else 1e12

    opts = dict(xatol=1e-9, fatol=1e-12, maxiter=400, maxfev=800)
    n_local = 12
    for i in range(n_local):
        if time.time() > deadline - 12.0:
            break
        if i == 0:
            x0 = best_c.copy()
        else:
            x0 = best_c + 0.15 * np.array([-1.13, 0.04, 0.0003086]) * (2.0 * rng.random(3) - 1.0)
        try:
            res = minimize(obj, x0, method="Nelder-Mead", options=opts)
            if res.fun < best:
                best = res.fun
                best_c = res.x
        except Exception:
            pass

    # --- Stage 3: high-precision Newton polish of the best point ------------
    try:
        v = _polish(best_c[0], best_c[1], best_c[2])
        if np.isfinite(v) and v < best:
            best = v
    except Exception:
        pass

    # safety: never return a non-finite / worse-than-seed result
    seed_val = _value(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)
    if not np.isfinite(best) or best > seed_val:
        best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
