"""Compact dark-energy growth-index fit (CPL cosmologies)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

_DEADLINE = 85.0  # safety margin under 90s wall-clock
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3.0)
    # DE density: (1-Om0)*a^(1+w0+wa)*exp(-wa*(a-1))
    Om_de = (1.0 - Om0) * np.exp((1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0))
    return Om_m + Om_de

def _Hprime_over_H(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3.0)
    Om_de = (1.0 - Om0) * np.exp((1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0))
    Om_tot = Om_m + Om_de
    dOm_dx = -3.0 * Om_m + Om_de * ((1.0 + w0 + wa) - wa * a)
    return 0.5 * dOm_dx / Om_tot

def _f_ode(x, y, Om0, w0, wa):
    a = math.exp(x)
    Om = _Om(np.array([a]), Om0, w0, wa)[0]
    H = _Hprime_over_H(np.array([a]), Om0, w0, wa)[0]
    f = y[0]
    # f' = (3/2) Om - f*(3/2 + H) - f^2
    return [(1.5) * Om - f * (1.5 + H) - f * f]

def _reference(Om0, w0, wa, a_grid):
    """Return f_ref and D_ref (normalized D(1)=1) on a_grid (ascending)."""
    x_grid = np.log(a_grid)
    x_min = x_grid[0]
    # integrate f from x_min to 0 with a small initial f
    f0 = 0.05
    sol = solve_ivp(_f_ode, [x_min, 0.0], [f0], args=(Om0, w0, wa),
                    t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-12, max_step=0.01)
    f_ref = sol.y[0]
    # D(a)/D(1) = exp(integral from 1 to a of f dx) = exp(-integral from a to 1 of f dx)
    # cumulative integral of f from x=0 (a=1) downward
    # integral_1^a f dx = - integral_a^1 f dx
    # use cumulative trapezoid from the end
    x = x_grid
    f = f_ref
    # integral from 0 to x_i  = - (integral from x_i to 0)
    # compute cumulative from last point (x=0) backward
    # I_i = integral_{x_i}^{x_{last}} f dx  (last is x=0)
    # build by cumulative trapezoid on reversed arrays
    fx = f[::-1]
    xx = x[::-1]
    # cumulative trapezoid
    dx = np.diff(xx)
    trap = 0.5 * (fx[1:] + fx[:-1]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    # cum[i] = integral from xx[i] to xx[-1]=x_min? No: cum[0]=0 at xx[0]=x=0
    # cum[i] = integral_{xx[0]=0}^{xx[i]} f dx  (reversed) = integral_0^{x_rev_i}
    # We want integral_{x_i}^{0} f dx = integral_0^{x_i} (-f) dx ... let's just do forward.
    # Simpler: integral_1^a f dx = integral_0^{x} f dx (x=ln a <=0). Compute forward cumulative trapezoid of f on x from x_min to 0, then value at x.
    dx_f = np.diff(x)
    trap_f = 0.5 * (f[1:] + f[:-1]) * dx_f
    cum_f = np.concatenate([[0.0], np.cumsum(trap_f)])
    # cum_f[i] = integral_{x_min}^{x_i} f dx
    # integral_0^{x_i} f dx = integral_{x_min}^{x_i} - integral_{x_min}^{0} = cum_f[i] - cum_f[-1]
    logD = cum_f - cum_f[-1]
    D_ref = np.exp(logD)
    return f_ref, D_ref

def _candidate(Om0, w0, wa, a_grid, c):
    a = np.asarray(a_grid, dtype=float)
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    f = Om**gamma
    # D(a)/D(1) = exp(-integral_ln a^0 f(exp x) dx) = exp(integral_1^a f dx)
    x = np.log(a)
    dx = np.diff(x)
    trap = 0.5 * (f[1:] + f[:-1]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    logD = cum - cum[-1]
    D = np.exp(logD)
    return f, D, gamma

def _model_grid():
    Om0 = np.array([0.20, 0.25, 0.30, 0.35])
    w0 = np.array([-1.5, -1.0, -0.5, 0.0])
    wa = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    models = []
    for o in Om0:
        for w in w0:
            for wv in wa:
                models.append((o, w, wv))
    return models

def _build_references(models, a_grid):
    refs = {}
    for (o, w, wv) in models:
        refs[(o, w, wv)] = _reference(o, w, wv, a_grid)
    return refs

def _objective(c, models, a_grid, refs, T):
    c = np.clip(c, -10.0, 10.0)
    rel_errs = []
    gamma_viol = 0.0
    for (o, w, wv) in models:
        f_ref, D_ref = refs[(o, w, wv)]
        f_c, D_c, gamma = _candidate(o, w, wv, a_grid, c)
        rel_f = np.abs(f_c - f_ref) / np.maximum(np.abs(f_ref), 1e-6)
        rel_D = np.abs(D_c - D_ref) / np.maximum(np.abs(D_ref), 1e-6)
        rel_errs.append(rel_f)
        rel_errs.append(rel_D)
        gamma_viol += np.sum(np.maximum(0.0, gamma - 1.2)) + np.sum(np.maximum(0.0, 0.1 - gamma))
    rel_errs = np.concatenate(rel_errs)
    # soft-max
    m = rel_errs.max()
    soft = m + (T / 100.0) * np.log(np.sum(np.exp((rel_errs - m) / (T / 100.0))))
    return soft + 50.0 * gamma_viol

def _exact_max(c, models, a_grid, refs):
    c = np.clip(c, -10.0, 10.0)
    worst = 0.0
    gv = 0.0
    for (o, w, wv) in models:
        f_ref, D_ref = refs[(o, w, wv)]
        f_c, D_c, gamma = _candidate(o, w, wv, a_grid, c)
        rel_f = np.abs(f_c - f_ref) / np.maximum(np.abs(f_ref), 1e-6)
        rel_D = np.abs(D_c - D_ref) / np.maximum(np.abs(D_ref), 1e-6)
        worst = max(worst, rel_f.max(), rel_D.max())
        gv += np.sum(np.maximum(0.0, gamma - 1.2)) + np.sum(np.maximum(0.0, 0.1 - gamma))
    return 100.0 * worst + 5000.0 * gv

def _fit(models, a_grid, refs, deadline):
    # global search
    bounds = [(-10, 10)] * 6
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = _exact_max(best_c, models, a_grid, refs)
    T = 5.0
    def obj(c):
        return _objective(c, models, a_grid, refs, T)
    # differential evolution (global)
    try:
        res = differential_evolution(obj, bounds, seed=1234, maxiter=40, popsize=8,
                                     tol=1e-8, polish=False, updating='immediate',
                                     init='sobol')
        if res.x is not None:
            c = np.clip(res.x, -10, 10)
            v = _exact_max(c, models, a_grid, refs)
            if v < best_val:
                best_val, best_c = v, c
    except Exception:
        pass
    if _time_left() < 15.0:
        return best_c, best_val
    # polish with L-BFGS-B on the soft objective, then exact
    try:
        r2 = minimize(obj, best_c, method='L-BFGS-B',
                      bounds=[(-10, 10)] * 6, options={'maxiter': 200, 'gtol': 1e-10})
        c = np.clip(r2.x, -10, 10)
        v = _exact_max(c, models, a_grid, refs)
        if v < best_val:
            best_val, best_c = v, c
    except Exception:
        pass
    if _time_left() < 8.0:
        return best_c, best_val
    # high-precision polish on exact (nonsmooth) via many small random restarts + local
    for _ in range(3):
        if _time_left() < 5.0:
            break
        try:
            r3 = minimize(lambda cc: _exact_max(cc, models, a_grid, refs), best_c,
                          method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10})
            c = np.clip(r3.x, -10, 10)
            v = _exact_max(c, models, a_grid, refs)
            if v < best_val:
                best_val, best_c = v, c
        except Exception:
            pass
    return best_c, best_val

def construct():
    global _START
    _START = time.time()
    models = _model_grid()
    a_grid = np.linspace(0.05, 1.0, 150)
    refs = _build_references(models, a_grid)
    c, val = _fit(models, a_grid, refs, _DEADLINE)
    # final safety: ensure gamma in range on a fine check; if violated, fall back to constant
    c = np.clip(np.asarray(c, dtype=float), -10.0, 10.0)
    # verify gamma range on validation grid (dense a)
    a_chk = np.linspace(0.02, 1.0, 400)
    ok = True
    for (o, w, wv) in models:
        _, _, gamma = _candidate(o, w, wv, a_chk, c)
        if gamma.min() < 0.1 - 1e-9 or gamma.max() > 1.2 + 1e-9:
            ok = False
            break
    if not ok:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    return [float(x) for x in c]
# EVOLVE-BLOCK-END
