"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as _np
import time as _time_mod
from scipy.optimize import minimize as _scipy_minimize
from scipy.integrate import solve_ivp as _solve_ivp

_DEADLINE = _time_mod.time() + 85.0

def _Om(a, Om0, w0, wa):
    Ode0 = 1.0 - Om0
    if abs(Ode0) < 1e-15:
        return Om0 * a**(-3)
    log_rho_m = _np.log(Om0) - 3.0*_np.log(a)
    log_rho_de = _np.log(Ode0) - 3.0*(1.0+w0-wa)*_np.log(a) + 3.0*wa*(1.0-a)
    m = _np.maximum(log_rho_m, log_rho_de)
    log_sum = m + _np.log(_np.exp(log_rho_m - m) + _np.exp(log_rho_de - m))
    return _np.exp(log_rho_m - log_sum)

def _gamma(c, a, Om0, w0, wa):
    Om_a = _Om(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_app(c, a, Om0, w0, wa):
    Om_a = _Om(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    return Om_a**gam

def _D_app(c, a_grid, Om0, w0, wa):
    f_vals = _f_app(c, a_grid, Om0, w0, wa)
    integrand = f_vals / a_grid
    idx_1 = _np.argmin(_np.abs(a_grid - 1.0))
    n = len(a_grid)
    integral = _np.zeros(n)
    for i in range(idx_1+1, n):
        integral[i] = integral[i-1] + 0.5*(integrand[i] + integrand[i-1])*(a_grid[i]-a_grid[i-1])
    for i in range(idx_1-1, -1, -1):
        integral[i] = integral[i+1] - 0.5*(integrand[i] + integrand[i+1])*(a_grid[i+1]-a_grid[i])
    return _np.exp(-integral)

def _gr_reference(a_grid, Om0, w0, wa):
    def rhs_D(x, y):
        a_val = _np.exp(x)
        Om_a_val = _Om(a_val, Om0, w0, wa)
        Ode_a_val = 1.0 - Om_a_val
        w_a_val = w0 + wa*(1.0 - a_val)
        Hprime_over_H = 1.5 * (Om_a_val + (1.0 + w_a_val) * Ode_a_val)
        D = y[0]
        Dp = y[1]
        Dpp = -Hprime_over_H * Dp + 1.5 * Om_a_val * D
        return [Dp, Dpp]
    f1 = 0.55 * Om0
    y0 = [1.0, f1]
    x_grid = _np.log(a_grid)
    sort_idx = _np.argsort(x_grid)
    x_sorted = x_grid[sort_idx]
    sol = _solve_ivp(rhs_D, [x_sorted[0], x_sorted[-1]], y0, t_eval=x_sorted,
                     method='RK45', rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        return None
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    f_vals = Dp_vals / D_vals
    idx_1 = _np.argmin(_np.abs(x_grid - 0.0))
    D_norm = D_vals / D_vals[idx_1]
    D_unsorted = _np.empty_like(D_norm)
    f_unsorted = _np.empty_like(f_vals)
    D_unsorted[sort_idx] = D_norm
    f_unsorted[sort_idx] = f_vals
    return f_unsorted, D_unsorted

def _gamma_valid(c, a_grid, cosmologies):
    for (Om0, w0, wa) in cosmologies:
        gam = _gamma(c, a_grid, Om0, w0, wa)
        if _np.any(gam < 0.1) or _np.any(gam > 1.2):
            return False
    return True

def construct():
    t_start = _time_mod.time()
    cosmologies = [
        (0.3, -1.0, 0.0), (0.3, -1.0, 0.5), (0.3, -1.0, -0.5),
        (0.3, -0.9, 0.0), (0.3, -0.9, 0.3), (0.3, -1.1, 0.0),
        (0.3, -1.1, 0.5), (0.25, -1.0, 0.0), (0.35, -1.0, 0.0),
        (0.3, -1.0, 1.0), (0.3, -1.0, -1.0), (0.3, -0.8, 0.0),
        (0.3, -1.2, 0.0),
    ]
    a_grid = _np.linspace(0.05, 1.0, 80)
    refs = []
    for (Om0, w0, wa) in cosmologies:
        ref = _gr_reference(a_grid, Om0, w0, wa)
        if ref is not None:
            refs.append((Om0, w0, wa, ref[0], ref[1]))
        if _time_mod.time() > _DEADLINE - 10.0:
            break
    def objective(c):
        if _time_mod.time() > _DEADLINE - 2.0:
            return 1e10
        if not _gamma_valid(c, a_grid, cosmologies):
            return 1e10
        try:
            max_err = 0.0
            for (Om0, w0, wa, f_gr, D_gr) in refs:
                f_ap = _f_app(c, a_grid, Om0, w0, wa)
                D_ap = _D_app(c, a_grid, Om0, w0, wa)
                err_f = _np.abs(f_ap - f_gr) / _np.maximum(_np.abs(f_gr), 1e-10)
                err_D = _np.abs(D_ap - D_gr) / _np.maximum(_np.abs(D_gr), 1e-10)
                max_err = max(max_err, float(_np.max(err_f)), float(_np.max(err_D)))
            return max_err
        except Exception:
            return 1e10
    c0 = _np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_err = objective(c0)
    rng = _np.random.default_rng(42)
    for i in range(40):
        if _time_mod.time() > _DEADLINE - 5.0:
            break
        if i == 0:
            c_init = c0.copy()
        else:
            c_init = c0 + rng.normal(0, 0.15, 6)
            c_init = _np.clip(c_init, -10, 10)
        try:
            result = _scipy_minimize(objective, c_init, method='Nelder-Mead',
                                     options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9,
                                              'adaptive': True})
            if _np.all(_np.abs(result.x) <= 10) and result.fun < best_err:
                best_c = result.x
                best_err = result.fun
        except Exception:
            pass
    if _time_mod.time() < _DEADLINE - 3.0:
        try:
            result = _scipy_minimize(objective, best_c, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                                              'adaptive': True})
            if _np.all(_np.abs(result.x) <= 10) and result.fun < best_err:
                best_c = result.x
                best_err = result.fun
        except Exception:
            pass
    best_c = _np.clip(best_c, -10, 10)
    return list(best_c)
# EVOLVE-BLOCK-END
