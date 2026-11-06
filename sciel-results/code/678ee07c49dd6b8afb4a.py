"""Cosmology: compact dark-energy growth approximation.

Fits six coefficients of a growth-index ansatz against a general-relativistic
second-order ODE reference, using Nelder-Mead with a wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---- Deadline guard: hard wall-clock limit with safety margin ----
_T_START = time.time()
_HARD_LIMIT = 70.0          # seconds; stay safely under the 90 s kill
_SOFT_LIMIT = 66.0          # stop *starting* new trials after this
_DEADLINE = _T_START + _HARD_LIMIT
_SOFT_DEADLINE = _T_START + _SOFT_LIMIT

def _time_left():
    return _DEADLINE - time.time()

def _soft_ok():
    return time.time() < _SOFT_DEADLINE

# ---- Cosmology / ansatz helpers (vectorised) ----
def _omega_m(a, Om0, wa):
    a = np.asarray(a, dtype=float)
    a3 = a ** 3
    return Om0 * a3 / (Om0 * a3 + (1.0 - Om0) * a ** (3.0 * (1.0 + wa)))

def _u_of(a, wa):
    a = np.asarray(a, dtype=float)
    return 1.0 + wa * (1.0 - a)

def _q_of(a, Om0, wa):
    return 1.0 - _omega_m(a, Om0, wa)

def _gamma(c, a, Om0, wa):
    a = np.asarray(a, dtype=float)
    q = _q_of(a, Om0, wa)
    u = _u_of(a, wa)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u
            + c[4] * wa * a + c[5] * u * q)

def _f(c, a, Om0, wa):
    a = np.asarray(a, dtype=float)
    g = _gamma(c, a, Om0, wa)
    return _omega_m(a, Om0, wa) ** g

def _D_ratio(c, a, Om0, wa):
    a = float(a)
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    x0 = math.log(a)

    def integrand(x):
        aa = math.exp(x)
        g = _gamma(c, np.array([aa]), Om0, wa)[0]
        return _omega_m(np.array([aa]), Om0, wa)[0] ** g

    try:
        I = solve_ivp(integrand, [x0, 0.0], [0.0], method="RK45",
                      rtol=1e-9, atol=1e-12, max_step=0.1)[0][-1]
    except Exception:
        I = 0.0
    return math.exp(-I)

# ---- GR reference (second-order ODE for the growth factor) ----
def _ref_f(a, Om0, wa):
    a = float(a)
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    Om = _omega_m(np.array([a]), Om0, wa)[0]
    E2 = Om + (1.0 - Om0) * a ** (3.0 * (1.0 + wa))
    Omd = 3.0 * Om * E2 - (3.0 / 2.0) * (1.0 - Om0) * (1.0 + wa) * a ** (3.0 * (1.0 + wa))
    return 0.5 * Om + 0.75 * Omd / E2

def _ref_D(a, Om0, wa):
    a = float(a)
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    # integrate y'' = 2 f y' - 3 H^2 y  backward from a=1, D(1)=1, D'(1)=f(1)
    def rhs(x, Y):
        aa = math.exp(x)
        Om = _omega_m(np.array([aa]), Om0, wa)[0]
        E2 = Om + (1.0 - Om0) * aa ** (3.0 * (1.0 + wa))
        ff = _ref_f(aa, Om0, wa)
        dY = [Y[1], 2.0 * ff * Y[1] - 3.0 * E2 * Y[0]]
        return dY
    x0 = math.log(a)
    try:
        sol = solve_ivp(rhs, [x0, 0.0], [1.0, _ref_f(1.0, Om0, wa)],
                        method="RK45", rtol=1e-9, atol=1e-12, max_step=0.1)
        return sol[0][0]
    except Exception:
        return 1.0

# ---- Validation cosmologies (disjoint public CPL models) ----
_MODELS = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.9, 0.5),
    (0.30, -1.1, 0.5),
    (0.30, -1.0, 1.0),
    (0.20, -1.0, 0.0),
    (0.30, -1.0, -1.0),
]
_A_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

# ---- Objective: worst relative error (f and D) across models & a ----
def _objective(c):
    c = np.asarray(c, dtype=float)
    if not np.all(np.isfinite(c)) or np.any(np.abs(c) > 10.0):
        return 1e6
    # gamma bounds at all checked locations
    for (Om0, w0, wa) in _MODELS:
        g = _gamma(c, _A_GRID, Om0, wa)
        if g.min() < 0.1 - 1e-9 or g.max() > 1.2 + 1e-9:
            return 1e6
    worst = 0.0
    for (Om0, w0, wa) in _MODELS:
        for a in _A_GRID:
            rf = _ref_f(a, Om0, wa)
            ff = _f(c, np.array([a]), Om0, wa)[0]
            err_f = abs(ff - rf) / max(abs(rf), 1e-9)
            rd = _ref_D(a, Om0, wa)
            dd = _D_ratio(c, a, Om0, wa)
            err_d = abs(dd - rd) / max(abs(rd), 1e-9)
            e = max(err_f, err_d)
            if e > worst:
                worst = e
    return worst

# ---- Seed (constant-gamma baseline) ----
_SEED = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

def _best_initial():
    """Return the best of the seed and a few analytic-ish starting points."""
    cands = [_SEED.copy(),
             np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
             np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
             np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0]),
             np.array([0.55, 0.0, 0.1, 0.0, 0.0, 0.0]),
             np.array([0.55, 0.1, 0.0, 0.0, 0.0, 0.0])]
    best = _SEED.copy()
    bestv = _objective(best)
    for c in cands:
        if not _soft_ok():
            break
        v = _objective(c)
        if v < bestv:
            bestv = v
            best = c.copy()
    return best, bestv

def construct():
    """Return six finite coefficients in [-10,10] minimising worst relative error."""
    bounds = [(-10.0, 10.0)] * 6

    # Stage 1: best starting point
    x0, f0 = _best_initial()
    best = x0.copy()
    bestv = f0

    # Stage 2: Nelder-Mead with deadline-aware callback
    def callback(xk, *args, **kwargs):
        v = _objective(xk)
        if v < bestv:
            nonlocal best, bestv
            best = np.asarray(xk, dtype=float).copy()
            bestv = v
        if _time_left() < 4.0:
            raise RuntimeError("deadline")

    if _time_left() > 15.0:
        try:
            res = minimize(_objective, x0, method="Nelder-Mead",
                           bounds=bounds,
                           options={"maxiter": 2000, "maxfev": 4000,
                                    "xatol": 1e-5, "fatol": 1e-8, "adaptive": True},
                           callback=callback)
            v = _objective(res.x)
            if v < bestv:
                best, bestv = np.asarray(res.x, dtype=float).copy(), v
        except Exception:
            pass

    # Stage 3: local polish from the best point found
    if _soft_ok() and _time_left() > 12.0:
        try:
            res = minimize(_objective, best, method="Nelder-Mead",
                           bounds=bounds,
                           options={"maxiter": 1000, "maxfev": 2000,
                                    "xatol": 1e-6, "fatol": 1e-9, "adaptive": True},
                           callback=callback)
            v = _objective(res.x)
            if v < bestv:
                best, bestv = np.asarray(res.x, dtype=float).copy(), v
        except Exception:
            pass

    # Enforce bounds / finiteness; fall back to seed if anything broke
    out = np.clip(np.asarray(best, dtype=float), -10.0, 10.0)
    if not np.all(np.isfinite(out)):
        out = _SEED.copy()
    # Final gamma-bounds sanity: if violated, fall back to the seed
    ok = True
    for (Om0, w0, wa) in _MODELS:
        g = _gamma(out, _A_GRID, Om0, wa)
        if g.min() < 0.1 - 1e-9 or g.max() > 1.2 + 1e-9:
            ok = False
            break
    if not ok:
        out = _SEED.copy()
    return [float(x) for x in out]
# EVOLVE-BLOCK-END
