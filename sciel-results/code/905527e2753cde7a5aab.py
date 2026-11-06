"""Compact dark-energy growth approximation: fit 6 coefficients to a GR reference."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_start_time = time.time()
_TIMEOUT = 85.0

_ref_cache = None

def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 1e-10, a, 1e-10)
    w_eff = 1.0 + w0 + wa
    de = (1.0 - Om0) * a_safe ** (-3.0 * w_eff) * np.exp(3.0 * wa * a_safe)
    m = Om0 * a_safe ** -3.0
    return m / (m + de)

def _gamma(a, Om0, w0, wa, c):
    c0, c1, c2, c3, c4, c5 = c
    Om_a = _Om(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f(a, Om0, w0, wa, c):
    Om_a = _Om(a, Om0, w0, wa)
    g = _gamma(a, Om0, w0, wa, c)
    return np.maximum(Om_a, 1e-12) ** g

def _D(a, Om0, w0, wa, c):
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    a_m = a[mask]
    a_min = np.min(a_m)
    n_pts = 800
    ln_grid = np.linspace(np.log(a_min), 0.0, n_pts)
    a_grid = np.exp(ln_grid)
    f_grid = _f(a_grid, Om0, w0, wa, c)
    cum = np.zeros_like(ln_grid)
    for i in range(1, n_pts):
        cum[i] = cum[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * (ln_grid[i] - ln_grid[i-1])
    ln_a_m = np.log(a_m)
    I_m = np.interp(ln_a_m, ln_grid, cum) - cum[-1]
    D[mask] = np.exp(I_m)
    return D

def _precompute():
    global _ref_cache
    if _ref_cache is not None:
        return _ref_cache

    models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.3, -1.5, 1.0),
        (0.3, -0.5, -0.5),
        (0.2, -1.2, 0.8),
        (0.4, -0.7, -0.3),
        (0.28, -1.0, 0.1),
        (0.32, -0.95, -0.15),
        (0.3, -1.0, 0.3),
        (0.3, -0.85, 0.25),
        (0.27, -1.1, 0.4),
        (0.33, -0.75, -0.25),
    ]
    a_vals = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                       0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ref_data = []

    for Om0, w0, wa in models:
        def ode_func(ln_a, y):
            f_val = y[0]
            a = np.exp(ln_a)
            Om_a = _Om(np.array([a]), Om0, w0, wa)[0]
            w_eff = 1.0 + w0 + wa
            g = a ** (-3.0 * w_eff) * np.exp(3.0 * wa * a)
            E2 = Om0 * a**(-3.0) + (1.0 - Om0) * g
            dE2_da = -3.0 * Om0 * a**(-4.0) + (1.0 - Om0) * g * 3.0 * (wa - w_eff / a)
            dlnE2_dlna = (dE2_da * a) / E2
            Hp_H = 0.5 * dlnE2_dlna
            f_prime = -1.5 * Om_a / E2 - (3.0 / a + Hp_H) * f_val - f_val**2
            return [f_prime]

        f_init = 0.55 * Om0**0.55
        ln_grid = np.log(a_vals)
        try:
            sol = solve_ivp(ode_func, [0, np.log(0.15)], [f_init], t_eval=ln_grid,
                            method='RK45', rtol=1e-10, atol=1e-12)
            if sol.success:
                f_ref = sol.y[0]
            else:
                f_ref = _f(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
        except Exception:
            f_ref = _f(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])

        ln_rev = ln_grid[::-1]
        f_rev = f_ref[::-1]
        cum = np.zeros_like(ln_rev)
        for i in range(1, len(ln_rev)):
            cum[i] = cum[i-1] + 0.5 * (f_rev[i] + f_rev[i-1]) * (ln_rev[i] - ln_rev[i-1])
        D_ref = np.exp(cum)[::-1]

        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))

    _ref_cache = ref_data
    return ref_data

def _score(coeffs):
    ref = _precompute()
    max_err = 0.0
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref:
        f_app = _f(a_vals, Om0, w0, wa, coeffs)
        D_app = _D(a_vals, Om0, w0, wa, coeffs)
        ef = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        ed = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(ef), np.max(ed))
        g = _gamma(a_vals, Om0, w0, wa, coeffs)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1.0
    return max_err

def _optimize():
    global _start_time
    deadline = _start_time + _TIMEOUT - 3.0
    rng = np.random.default_rng(12345)

    _precompute()

    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _score(best_coeffs)

    # Multi-start Nelder-Mead with diverse initial points
    starts = [
        np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03]),
        np.array([0.5, 0.1, -0.05, 0.1, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.1, 0.0, 0.0, 0.05]),
        np.array([0.55, 0.05, 0.0, 0.05, -0.05, 0.0]),
        np.array([0.5, 0.0, 0.1, -0.1, 0.05, -0.05]),
        np.array([0.55, -0.05, 0.05, 0.0, 0.0, 0.05]),
        np.array([0.6, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.0, 0.05, 0.05]),
    ]

    for start in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_score, start, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Random restarts
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        c = rng.normal(
            loc=best_coeffs,
            scale=[0.1, 0.3, 0.1, 0.3, 0.3, 0.2]
        )
        c = np.clip(c, -10, 10)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_coeffs = c.copy()
            # Quick local polish
            try:
                res = minimize(_score, best_coeffs, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = np.clip(res.x, -10, 10)
            except Exception:
                pass

    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(_score, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    return best_coeffs

_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize()
    return _optimized_coeffs
