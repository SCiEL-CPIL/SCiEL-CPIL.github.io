"""Compact dark-energy growth-index fit; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

_DEADLINE = 82.0  # safety margin below the 90 s wall-clock kill


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
    f = 0.5 * Om + 0.6 * np.sqrt(Om * (1.0 - Om)) + 0.2 * dOm / (3.0 * a)
    return f


def _D_gr(a, Om0, w0, wa, n=1400):
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


def _D_app(c, a, Om0, w0, wa, n=1400):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + fa[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _eval(c, models, a_eval, a_fit):
    err_f = 0.0
    err_D = 0.0
    for (Om0, w0, wa) in models:
        if c[0] < 0.1 - 2.0 or c[0] > 1.2 + 2.0:
            return 1e9
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = _f_gr(a_eval, Om0, w0, wa)
        err_f = max(err_f, float(np.max(np.abs((fa - fg) / fg))))
        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        err_D = max(err_D, float(np.max(np.abs((Da - Dg) / Dg))))
        # penalty if gamma leaves [0.1,1.2] on the fit grid
        g = _gamma(c, a_fit, Om0, w0, wa)
        pen = np.maximum(0.0, 0.1 - g) + np.maximum(0.0, g - 1.2)
        if pen.max() > 1e-3:
            return 1e9
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
    a_eval = np.logspace(-2.0, 0.0, 60)
    a_fit = np.logspace(-2.0, 0.0, 40)

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

    # coarse multi-start local search
    rng = np.random.default_rng(12345)
    starts = [np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])]
    for _ in range(30):
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]) + rng.normal(0.0, 0.25, 6)
        c = np.clip(c, -10.0, 10.0)
        starts.append(c)

    for c in starts:
        if _now() > _DEADLINE:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: _eval(np.clip(x, -10.0, 10.0), models, a_eval, a_fit),
                np.clip(c, -10.0, 10.0),
                method="Nelder-Mead",
                options={"maxiter": 400, "maxfev": 600,
                         "xatol": 1e-5, "fatol": 1e-6},
            )
            _try(res.x)
        except Exception:
            pass

    # final clamp and validation
    best_c = np.clip(best_c, -10.0, 10.0)
    for (Om0, w0, wa) in models:
        g = _gamma(best_c, a_fit, Om0, w0, wa)
        if g.min() < 0.1 or g.max() > 1.2:
            best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            break
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
