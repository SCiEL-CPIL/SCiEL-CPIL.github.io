"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_start_time = time.time()
_DEADLINE = 55.0

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def _compute_om_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    ed_term = (1.0 - Om0) * np.power(a_safe, 3.0*(1.0+w0)) * np.exp(3.0*wa*(a_safe - 1.0))
    om_term = Om0 * np.power(a_safe, -3.0)
    return om_term / (om_term + ed_term)

def _solve_reference(Om0, w0, wa, a_grid):
    a_grid = np.asarray(a_grid, dtype=float)
    
    def _dlnH_dlna(a_val):
        a_val = max(float(a_val), 1e-10)
        e2 = Om0 * a_val**(-3) + (1-Om0) * a_val**(3*(1+w0)) * np.exp(3*wa*(a_val-1))
        de2_dlna = -3*Om0*a_val**(-3) + (1-Om0)*3*(1+w0+wa)*a_val**(3*(1+w0))*np.exp(3*wa*(a_val-1))
        return 0.5 * de2_dlna / e2
    
    def ode_rhs(x, y):
        a_val = np.exp(x)
        om_a_val = float(_compute_om_a(a_val, Om0, w0, wa)[0])
        dlnH = _dlnH_dlna(a_val)
        D, f = y
        return [f * D, -f**2 - (1.5 + dlnH)*f + 3*om_a_val/2]
    
    x_min = np.log(max(float(a_grid[0]), 1e-4))
    x_eval = np.log(a_grid)
    
    a_init = np.exp(x_min)
    om_init = float(_compute_om_a(a_init, Om0, w0, wa)[0])
    f_init = om_init**0.55
    
    sol = solve_ivp(ode_rhs, [x_min, 0.0], [1.0, f_init], 
                    method='RK45', rtol=1e-8, atol=1e-10, t_eval=x_eval)
    
    if sol.success:
        D_ref = sol.y[0] / sol.y[0, -1]
        f_ref = sol.y[1]
    else:
        f_ref = np.array([float(_compute_om_a(a, Om0, w0, wa)[0])**0.55 for a in a_grid])
        D_ref = np.ones_like(a_grid)
    
    return f_ref, D_ref

_cosmologies = [(0.3, -1.0, 0.0), (0.3, -0.9, 0.3), (0.3, -1.1, 0.5), (0.25, -1.0, 0.0), (0.35, -0.8, 0.2)]
_a_grid = np.linspace(0.05, 0.95, 15)
_ref_data = []
for _Om0, _w0, _wa in _cosmologies:
    if _check_deadline():
        break
    _f, _D = _solve_reference(_Om0, _w0, _wa, _a_grid)
    _ref_data.append((_Om0, _w0, _wa, _f, _D))

def _f_approx_vec(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    om_a = _compute_om_a(a, Om0, w0, wa)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    return np.power(np.maximum(om_a, 1e-10), gamma)

def _D_approx_vec(coeffs, Om0, w0, wa, a_grid):
    D_approx = np.zeros_like(a_grid)
    for i, a_val in enumerate(a_grid):
        if a_val >= 1.0:
            D_approx[i] = 1.0
        else:
            x_min = np.log(a_val)
            x_g = np.linspace(x_min, 0.0, 30)
            f_g = _f_approx_vec(np.exp(x_g), Om0, w0, wa, coeffs)
            D_approx[i] = np.exp(-np.trapz(f_g, x_g))
    return D_approx

def _objective(coeffs):
    if _check_deadline():
        return np.inf
    max_err = 0.0
    for Om0, w0, wa, f_ref, D_ref in _ref_data:
        f_app = _f_approx_vec(_a_grid, Om0, w0, wa, coeffs)
        D_app = _D_approx_vec(coeffs, Om0, w0, wa, _a_grid)
        fe = np.max(np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10))
        De = np.max(np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10))
        max_err = max(max_err, fe, De)
        if _check_deadline():
            return np.inf
    return max_err

def _fit():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10)]*6
    best = x0
    best_obj = _objective(x0)
    
    if not _check_deadline():
        try:
            r = minimize(_objective, x0, method='Nelder-Mead', bounds=bounds,
                        options={'maxiter': 200, 'adaptive': True})
            if r.fun < best_obj:
                best, best_obj = r.x, r.fun
        except:
            pass
    
    if not _check_deadline():
        try:
            r2 = minimize(_objective, best, method='Powell', bounds=bounds,
                         options={'maxiter': 30})
            if r2.fun < best_obj:
                best, best_obj = r2.x, r2.fun
        except:
            pass
    
    best = np.clip(np.nan_to_num(best, nan=0.55), -10, 10)
    return list(best)

def construct():
    if _check_deadline():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    c = _fit()
    if _check_deadline():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    return c
# EVOLVE-BLOCK-END

