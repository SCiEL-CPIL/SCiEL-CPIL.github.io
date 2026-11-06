"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

_GR_CACHE = {}

def _Om_m(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Ed2 = Om0 * a**3 + (1.0 - Om0) * np.exp(3.0 * wa * (1.0 - a))
    return Om0 * a**3 / Ed2

def _solve_f_gr(Om0, w0, wa, a_grid):
    key = (round(Om0, 6), round(w0, 6), round(wa, 6), tuple(np.round(a_grid, 10)))
    if key in _GR_CACHE:
        return _GR_CACHE[key]
    
    a_grid = np.sort(np.asarray(a_grid, dtype=float))
    x_grid = np.log(a_grid)
    
    def ode_fun(x, y):
        a = np.exp(x)
        f = y[0]
        Om_m_val = _Om_m(a, Om0, w0, wa)
        return [1.5 * Om_m_val - f * f - f]
    
    x_start = np.log(1e-4)
    x_end = 0.0
    x_eval = np.clip(x_grid, x_start, x_end)
    
    sol = solve_ivp(ode_fun, [x_start, x_end], [1.0], t_eval=x_eval,
                    rtol=1e-10, atol=1e-12, max_step=0.005)
    
    if not sol.success:
        result = np.full_like(a_grid, 0.5 * Om0**0.55)
    else:
        result = sol.y[0]
    
    _GR_CACHE[key] = result
    return result

def _compute_D_from_f(f_grid, a_grid):
    a_grid = np.asarray(a_grid, dtype=float)
    f_grid = np.asarray(f_grid, dtype=float)
    x_grid = np.log(a_grid)
    
    D_vals = np.ones_like(a_grid)
    for i, ai in enumerate(a_grid):
        if ai >= 1.0:
            D_vals[i] = 1.0
        else:
            x_lo = np.log(ai)
            mask = (x_grid >= x_lo) & (x_grid <= 0.0)
            if np.sum(mask) < 2:
                D_vals[i] = np.exp(f_grid[i] * (0.0 - x_lo))
            else:
                x_int = x_grid[mask]
                f_int = f_grid[mask]
                if x_int[-1] < 0:
                    x_int = np.append(x_int, 0.0)
                    f_int = np.append(f_int, f_grid[np.argmin(np.abs(x_grid))])
                integral = np.trapz(f_int, x_int)
                D_vals[i] = np.exp(integral)
    
    return D_vals

def _compute_gamma(c, a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m_val = _Om_m(a, Om0, w0, wa)
    q = 1.0 - Om_m_val
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _compute_f_approx(c, a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m_val = _Om_m(a, Om0, w0, wa)
    gamma = _compute_gamma(c, a, Om0, w0, wa)
    return Om_m_val ** gamma

_a_grid = np.linspace(0.1, 1.0, 15)
_test_cosmos = [
    (0.27, -1.0, 0.0),
    (0.30, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.8, 0.0),
    (0.30, -1.2, 0.0),
    (0.30, -1.0, 0.3),
]
_GR_REF = {}
for _Om0, _w0, _wa in _test_cosmos:
    try:
        _f_ref = _solve_f_gr(_Om0, _w0, _wa, _a_grid)
        _D_ref = _compute_D_from_f(_f_ref, _a_grid)
        _GR_REF[(_Om0, _w0, _wa)] = (_f_ref, _D_ref)
    except Exception:
        pass

def _objective(c):
    c = np.clip(c, -10, 10)
    
    a_test = np.linspace(0.1, 1.0, 20)
    for Om0, w0, wa in [(0.25, -1.0, 0.0), (0.35, -1.0, 0.0), (0.30, -0.8, 0.3)]:
        gamma_test = _compute_gamma(c, a_test, Om0, w0, wa)
        if np.any(gamma_test < 0.1) or np.any(gamma_test > 1.2):
            return 10.0
    
    max_err = 0.0
    for key, (f_ref, D_ref) in _GR_REF.items():
        Om0, w0, wa = key
        f_our = _compute_f_approx(c, _a_grid, Om0, w0, wa)
        D_our = _compute_D_from_f(f_our, _a_grid)
        
        rel_f = np.abs(f_our - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        rel_D = np.abs(D_our - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        err = np.max(np.concatenate([rel_f, rel_D]))
        max_err = max(max_err, err)
        if time.time() > _DEADLINE - _MARGIN:
            break
    
    return max_err

def construct():
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6
    
    best_c = c0.copy()
    best_err = _objective(best_c)
    
    try:
        res = minimize(_objective, c0, method='Nelder-Mead',
                      options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_err:
            best_err = res.fun
            best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass
    
    if time.time() > _DEADLINE - _MARGIN:
        return [float(x) for x in best_c]
    
    try:
        res = minimize(_objective, best_c, method='Powell', bounds=bounds,
                      options={'maxiter': 80, 'ftol': 1e-7})
        if res.fun < best_err:
            best_err = res.fun
            best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass
    
    best_c = np.clip(best_c, -10, 10)
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
