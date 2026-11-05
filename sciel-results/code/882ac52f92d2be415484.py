"""Compact dark-energy growth-index fit against a GR second-order ODE reference."""
import math
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 90.0
_MARGIN = 3.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _eval_model(c, Om0, w0, wa, a_grid):
    a = np.asarray(a_grid, float)
    Om = Om0 * a ** -3
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    g = np.clip(g, 0.1, 1.2)
    f = Om ** g
    F = np.exp(-np.log(a))
    lnD = np.array([
        -np.trapezoid(np.log(F[1:]), F[1:]) for F in F
    ])
    Dn = np.exp(lnD)
    return f, Dn

def _reference(Om0, w0, wa, a_grid):
    a = np.asarray(a_grid, float)
    Om = Om0 * a ** -3
    w = w0 + wa * (1.0 - a)
    H2 = Om + (1.0 - Om) * a ** (-3.0 * (1.0 + w))

    def rhs(x, y):
        O = Om0 * x ** -3
        W = w0 + wa * (1.0 - x)
        h2 = O + (1.0 - O) * x ** (-3.0 * (1.0 + W))
        dh2 = -3.0 * O * (1.0 + W)
        p = 0.5 * dh2 / h2
        return [y[1], 1.5 * O - p * y[1] - 0.5 * y[0]]

    sol = solve_ivp(rhs, (1.0, 1e-3), [1.0, 1.5 * Om0], t_eval=a,
                    method="DOP853", rtol=1e-10, atol=1e-12)
    D = sol.y[0]
    Dn = D / D[0]
    f = 1.5 * Om - 0.5 * (sol.y[1] / D)
    return f, Dn

def _objective(c, Om0, w0, wa, a_grid, f_ref, D_ref):
    f, Dn = _eval_model(c, Om0, w0, wa, a_grid)
    rf = np.max(np.abs((f - f_ref) / f_ref))
    rd = np.max(np.abs((Dn - D_ref) / D_ref))
    return 100.0 * max(rf, rd)

def _seed():
    return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def construct():
    if _time_left() <= 0.5:
        return [float(v) for v in _seed()]

    # Validation-like cosmologies (disjoint from any hidden set; use a spread).
    models = [(0.30, -1.0, 0.0), (0.28, -0.9, 0.3), (0.32, -1.0, 0.5),
              (0.25, -1.1, 0.4), (0.35, -0.8, -0.3), (0.30, -1.2, 0.6)]

    a_grid = np.geomspace(0.1, 1.0, 200)
    a_grid[0] = 0.1

    # Precompute references.
    refs = []
    for (Om0, w0, wa) in models:
        if _time_left() <= 2.0:
            break
        try:
            fr, dr = _reference(Om0, w0, wa, a_grid)
            if np.all(np.isfinite(fr)) and np.all(np.isfinite(dr)) and np.all(dr > 0):
                refs.append((Om0, w0, wa, fr, dr))
        except Exception:
            pass

    if not refs:
        return [float(v) for v in _seed()]

    best_c = _seed().copy()
    best_val = None
    for (Om0, w0, wa, fr, dr) in refs:
        if _time_left() <= 1.5:
            break
        def obj(c):
            return _objective(c, Om0, w0, wa, a_grid, fr, dr)
        # Multi-start local optimization.
        starts = [_seed(),
                  np.array([0.55, 0.2, 0.0, -0.5, 0.0, 0.0]),
                  np.array([0.55, -0.1, 0.1, -0.5, 0.1, 0.0]),
                  np.array([0.55, 0.0, 0.0, -0.4, 0.0, 0.1])]
        for s in starts:
            if _time_left() <= 1.0:
                break
            try:
                res = minimize(obj, s, method="L-BFGS-B",
                               bounds=[(0.1, 10.0)]*6,
                               options={"maxiter": 200, "ftol": 1e-12})
                val = res.fun
                if np.isfinite(val) and (best_val is None or val < best_val):
                    best_val = val
                    best_c = res.x.copy()
            except Exception:
                pass
        if best_val is None:
            best_val = obj(_seed())
            best_c = _seed().copy()

    # Clip coefficients to required range.
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(v) for v in best_c]
# EVOLVE-BLOCK-END
