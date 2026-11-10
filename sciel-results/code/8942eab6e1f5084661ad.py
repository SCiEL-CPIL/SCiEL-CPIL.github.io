"""Rewritten program: least-squares fit of 6-coefficient growth ansatz to numerical ODE reference."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 85.0

def _check_deadline():
    return (time.time() - _START_TIME) < _DEADLINE

def _Om_m(a, Om0, w0, wa):
    Om_de0 = 1.0 - Om0
    rho_m = Om0 * a**(-3)
    rho_de = Om_de0 * a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(a-1.0))
    rho_tot = rho_m + rho_de
    return rho_m / rho_tot

def _solve_growth_ode(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=80):
    Om_de0 = 1.0 - Om0
    
    def dlnOm_dx(x):
        a = np.exp(x)
        da = 1e-7
        Om_c = _Om_m(a, Om0, w0, wa)
        Om_p = _Om_m(a + da, Om0, w0, wa)
        Om_mn = _Om_m(a - da, Om0, w0, wa)
        dOm_da = (Om_p - Om_mn) / (2*da)
        return (dOm_da / Om_c) * a
    
    def ode_rhs(x, y):
        D, Dp = y
        a = np.exp(x)
        Om = _Om_m(a, Om0, w0, wa)
        dlnOm_dx = dlnOm_dx(x)
        Dpp = -(1.5 + 0.5*dlnOm_dx)*Dp - 1.5*Om*D
        return [Dp, Dpp]
    
    x_min = np.log(a_min)
    x_max = 0.0
    f1 = Om0**0.5
    y0 = [1.0, f1]
    
    sol = solve_ivp(ode_rhs, [x_min, x_max], 
                    [None, None], method='RK45', rtol=1e-10, atol=1e-12)
    # We need BCs at x_min. Let's integrate forward from x_min with guessed BCs,
    # then adjust so that D(1)=1.
    
    # Simpler: integrate from x_max backward with D(1)=1, f(1)=Om0^0.5
    sol_b = solve_ivp(ode_rhs, [x_max, x_min], y0, method='RK45', 
                      rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol_b.success:
        return None, None, None
    
    # Now integrate forward from x_min using the BCs from sol_b
    D_end = sol_b.y[0, -1]
    Dp_end = sol_b.y[1, -1]
    
    sol_f = solve_ivp(ode_rhs, [x_min, x_max], [D_end, Dp_end], 
                      method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol_f.success:
        return None, None, None
    
    a_vals = np.linspace(a_min, a_max, n_points)
    x_vals = np.log(a_vals)
    
    D_vals = sol_f.sol(x_vals)[0]
    Dp_vals = sol_f.sol(x_vals)[1]
    
    # Normalize D(1) = 1
    D_at_1 = D_vals[-1]
    D_vals = D_vals / D_at_1
    Dp_vals = Dp_vals / D_at_1
    f_vals = Dp_vals / D_vals
    
    return a_vals, f_vals, D_vals

def _compute_error(coeffs, ref_data):
    max_err = 0.0
    for (a_vals, f_ref, D_ref, Om0, w0, wa) in ref_data:
        Om_vals = _Om_m(a_vals, Om0, w0, wa)
        q_vals = 1.0 - Om_vals
        u_vals = 1.0 + w0 + wa*(1.0 - a_vals)
        
        c0, c1, c2, c3, c4, c5 = coeffs
        gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
        
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e6
        
        f_our = Om_vals**gamma_vals
        err_f = np.abs(f_our - f_ref) / np.abs(f_ref)
        max_err = max(max_err, np.max(err_f))
        
        # Normalized D
        x_grid = np.log(a_vals)
        dx = np.diff(x_grid)
        avg_f = 0.5 * (f_our[:-1] + f_our[1:])
        cum_int = np.concatenate([[0], np.cumsum(avg_f * dx)])
        int_to_0 = cum_int[-1] - cum_int
        D_our_norm = np.exp(-int_to_0)
        
        err_D = np.abs(D_our_norm - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_D))
        
        if max_err > 0.15:
            break
    
    return max_err

# Pre-compute reference data
_cosmologies = [
    (0.30, -1.0, 0.0),
    (0.28, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.9, 0.0),
    (0.25, -1.0, 0.3),
    (0.35, -1.1, 0.2),
]

_ref_data = []
for (Om0, w0, wa) in _cosmologies:
    if not _check_deadline():
        break
    try:
        a_vals, f_vals, D_vals = _solve_growth_ode(Om0, w0, wa)
        if a_vals is not None:
            _ref_data.append((a_vals, f_vals, D_vals, Om0, w0, wa))
    except Exception:
        pass

# Fit coefficients
_best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
if _ref_data and _check_deadline():
    _best_err = _compute_error(_best_coeffs, _ref_data)
    
    _initial_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.55, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.50, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
    
    for x0 in _initial_points:
        if not _check_deadline():
            break
        try:
            result = minimize(_compute_error, x0, args=(_ref_data,), 
                            method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < _best_err:
                _best_err = result.fun
                _best_coeffs = list(result.x)
        except Exception:
            pass

_best_coeffs = [max(-10.0, min(10.0, float(c))) for c in _best_coeffs]

def construct():
    return _best_coeffs
