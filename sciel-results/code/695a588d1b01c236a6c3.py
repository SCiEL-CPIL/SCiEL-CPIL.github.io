"""Compact dark-energy growth-index fit; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 85.0  # safety margin below the 90 s wall-clock kill


def _now():
    return time.time()


def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    wa_a = 1.0 - a
    w = w0 + wa * wa_a
    x = np.maximum(a, 1e-300)
    num = 1.0 + w
    den = 1.0 + 3.0 * w
    E2 = Om0 * np.power(x, 3.0 * den) + (1.0 - Om0) * np.power(x, -3.0 * num)
    return Om0 * np.power(x, 3.0) / E2


def _f_gr(a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    dOm = (3.0 * Om - 1.5 * Om * (w0 + wa * (1.0 - a))) / a
    f = 0.5 * Om + 0.6 * np.sqrt(np.maximum(Om * (1.0 - Om), 0.0)) + 0.2 * dOm / (3.0 * a)
    return f


def _D_gr(a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    f = _f_gr(np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _gamma(c, a, Om0, w0, wa):
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _f_app(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    return np.power(np.maximum(Om, 1e-300), g)


def _D_app(c, a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _eval(c, models, a_eval, a_fit):
    # Check gamma bounds
    for (Om0, w0, wa) in models:
        g = _gamma(c, a_fit, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e9
        if not np.all(np.isfinite(g)):
            return 1e9
    
    err_f = 0.0
    err_D = 0.0
    for (Om0, w0, wa) in models:
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = _f_gr(a_eval, Om0, w0, wa)
        err_f = max(err_f, float(np.max(np.abs((fa - fg) / np.maximum(fg, 1e-10)))))
        
        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        err_D = max(err_D, float(np.max(np.abs((Da - Dg) / np.maximum(Dg, 1e-10)))))
    
    return 100.0 * max(err_f, err_D)


def construct():
    t0 = _now()
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 1.0),
        (0.30, -1.0, 0.5),
        (0.27, -1.0, 0.3),
        (0.30, -0.9, 0.0),
    ]
    a_eval = np.logspace(-2.0, 0.0, 80)
    a_fit = np.logspace(-2.0, 0.0, 60)

    # Initial guesses based on known growth index approximations
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_v = _eval(best_c, models, a_eval, a_fit)

    def _try(c):
        nonlocal best_c, best_v
        if not np.all(np.isfinite(c)) or np.any(np.abs(c) > 10.0):
            return
        v = _eval(c, models, a_eval, a_fit)
        if v < best_v:
            best_v = v
            best_c = c.copy()

    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])]
    
    # Add perturbations around the seed
    for _ in range(50):
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]) + rng.normal(0.0, 0.2, 6)
        c = np.clip(c, -10.0, 10.0)
        starts.append(c)
    
    # Add some structured starts based on known approximations
    structured_starts = [
        np.array([0.55, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
        np.array([0.55, 0.05, 0.05, 0.05, 0.05, 0.05]),
        np.array([0.55, -0.05, -0.05, -0.05, -0.05, -0.05]),
    ]
    starts.extend(structured_starts)

    # Local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    for i, c in enumerate(starts):
        if _now() > _DEADLINE:
            break
        try:
            res = minimize(
                lambda x: _eval(np.clip(x, -10.0, 10.0), models, a_eval, a_fit),
                np.clip(c, -10.0, 10.0),
                method="Nelder-Mead",
                options={"maxiter": 300, "maxfev": 400,
                         "xatol": 1e-6, "fatol": 1e-7},
            )
            _try(res.x)
        except Exception:
            pass

    # Final refinement with more iterations
    if _now() < _DEADLINE - 5.0:
        try:
            res = minimize(
                lambda x: _eval(np.clip(x, -10.0, 10.0), models, a_eval, a_fit),
                np.clip(best_c, -10.0, 10.0),
                method="Nelder-Mead",
                options={"maxiter": 500, "maxfev": 600,
                         "xatol": 1e-7, "fatol": 1e-8},
            )
            _try(res.x)
        except Exception:
            pass

    # Final clamp and validation
    best_c = np.clip(best_c, -10.0, 10.0)
    for (Om0, w0, wa) in models:
        g = _gamma(best_c, a_fit, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            break
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
