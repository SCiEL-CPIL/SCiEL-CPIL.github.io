# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Physicists' Hermite polynomials (roots at 0, 4, 8, 12)
_H0 = np.ones(1)
_H4 = np.array([3.0, 0.0, -12.0, 0.0, 8.0])
_H8 = np.array([10.0, 0.0, -48.0, 0.0, 336.0, 0.0, -448.0, 0.0, 128.0])
_H12 = np.array([105.0, 0.0, -360.0, 0.0, 7560.0, 0.0, -30240.0, 0.0,
                 33600.0, 0.0, -13440.0, 0.0, 2048.0])

# Fixed evaluation grid (fine enough to capture the quartic/gaussian shape)
_X = np.linspace(-6.0, 6.0, 801)
_W = np.exp(-_X * _X / 2.0)
_P0 = _H0
_P4 = np.polyval(_H4, _X)
_P8 = np.polyval(_H8, _X)
_P12 = np.polyval(_H12, _X)

# Precompute monomial moments of the weight (for the derivative identity c=4a-2b)
_M0 = np.sum(_W)
_M2 = np.sum(_X * _X * _W)
_M4 = np.sum(_X * _X * _X * _X * _W)
_M6 = np.sum(_X * _X * _X * _X * _X * _X * _W)


def _coeffs(c):
    c0, c4, c8 = c
    c12 = -(c0 + c4 * 3.0 + c8 * 10.0) / 105.0
    return c0, c4, c8, c12


def _objective(c):
    c0, c4, c8, c12 = _coeffs(c)
    g = (c0 * _P0 + c4 * _P4 + c8 * _P8 + c12 * _P12) * _W
    a = np.sum(g * g)
    if a <= 1e-300:
        return 1e30
    b = np.sum(_X * _X * g * g)
    if b <= 1e-300:
        return 1e30
    d = np.sum((c0 * _P0 + c4 * _P4 + c8 * _P8 + c12 * _P12) ** 2)
    cder = 4.0 * a - 2.0 * b
    return ((2.0 * b / a + 0.5 * a / b) * (cder / (4.0 * a) + d / (4.0 * b)))


def construct():
    deadline = time.time() + 180.0
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _objective(seed)

    # Phase 1: global grid with explicit deadline guard
    n = 100
    lo, hi = -1.0, 1.0
    for i in range(n + 1):
        if time.time() > deadline - 6.0:
            break
        c0 = lo + (hi - lo) * i / n
        for j in range(n + 1):
            if time.time() > deadline - 6.0:
                break
            c4 = lo + (hi - lo) * j / n
            for k in range(n + 1):
                if time.time() > deadline - 6.0:
                    break
                c8 = lo + (hi - lo) * k / n
                cand = np.array([c0, c4, c8])
                v = _objective(cand)
                if v < best_val:
                    best_val = v
                    best = cand

    # Phase 2: Nelder-Mead polish from the best grid point
    try:
        if time.time() < deadline - 3.0:
            res = minimize(_objective, best, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12})
            if res.fun < best_val:
                best = np.asarray(res.x, dtype=float)
                best_val = float(res.fun)
    except Exception:
        pass

    # Phase 3: high-precision polish pass (tighter tol, more iterations)
    try:
        if time.time() < deadline - 1.0:
            res = minimize(_objective, best, method="Nelder-Mead",
                           options={"maxiter": 60000, "xatol": 1e-12, "fatol": 1e-15,
                                    "adaptive": True})
            if res.fun < best_val:
                best = np.asarray(res.x, dtype=float)
                best_val = float(res.fun)
    except Exception:
        pass

    # Safety: if the polish drifted or failed, fall back to the best grid point
    if not np.all(np.isfinite(best)):
        best = seed
    return best
# EVOLVE-BLOCK-END
