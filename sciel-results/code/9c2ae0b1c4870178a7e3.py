"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def _compute_om_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    de_exponent = -3.0 * (1.0 + w0 + wa)
    if a >= 1.0:
        de_factor = 1.0
    else:
        one_minus_a = max(1.0 - a, 1e-15)
        de_factor = one_minus_a ** de_exponent
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    Om_a = (Om0 / (a ** 3)) / E2
    return Om_a

def _solve_growth_ode(Om0, w0, wa, a_min=0.01, a_max=1.0, n_points=300):
    """
    Solve the GR growth ODE in terms of x = ln(a):
    D'' + (3/2)Om(a) D' - (3/2)Om(a) D = 0
    where ' denotes d/dx.
    Initial conditions: D(a_min) = a_min, f(a_min) = 1.
    Returns x_arr, D_norm (normalized so D(1)=1), f_arr.
    """
    x_min = np.log(a_min)
    x_max = np.log(a_max)
    
    def ode_func(x, y):
        D, f = y
        a = np.exp(x)
        Om_a = _compute_om_a(a, Om0, w0, wa)
        D_prime = f * D
        f_prime = -f**2 - 1.5 * Om_a * f + 1.5 * Om_a
        return [D_prime, f_prime]
    
    y0 = [a_min, 1.0]
    x_eval = np.linspace(x_min, x_max, n_points)
    
    sol = solve_ivp(ode_func, [x_min, x_max], y0, t_eval=x_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=0.005)
    
    if not sol.success:
        return None, None, None
    
    x_arr = sol.t
    D_arr = sol.y[0]
    f_arr = sol.y[1]
    
    D_norm = D_arr / D_arr[-1]
    
    return x_arr, D_norm, f_arr

def _compute_reference(Om0, w0, wa, a_values):
    """Compute reference f and D/D(1) for given cosmology and a values."""
    a_min = max(0.01, a_values.min() * 0.5)
    a_max = 1.0
    
    x_arr, D_arr, f_arr = _solve_growth_ode(Om0, w0, wa, a_min, a_max, n_points=400)
    
    if x_arr is None:
        return None, None
    
    a_eval = np.array(a_values)
    x_eval = np.log(a_eval)
    
    f_ref = np.interp(x_eval, x_arr, f_arr)
    D_ref = np.interp(x_eval, x_arr, D_arr)
    
    return f_ref, D_ref

def _approx_D_ratio(a, Om0, w0, wa, coeffs):
    """Compute D(a)/D(1) using the approximate f = Om(a)^gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    if a >= 1.0:
        return 1.0
    
    # Integrate f(a')/a' da' from a to 1
    n_int = 60
    a_grid = np.linspace(a, 1.0, n_int)
    f_vals = np.zeros(n_int)
    for j, aj in enumerate(a_grid):
        Om_a = _compute_om_a(aj, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - aj)
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * aj + c5 * u * q
        if gamma < 0.1 or gamma > 1.2:
            return -1.0  # signal invalid
        f_vals[j] = Om_a ** gamma
    
    integrand = f_vals / a_grid
    integral = np.trapz(integrand, a_grid)
    return np.exp(-integral)

def _growth_index_error(coeffs):
    """Compute the worst relative error for a given set of 6 coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, -0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.8, 0.0),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.0, -1.0),
        (0.3, -1.0, 1.0),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5),
    ]
    
    a_values = np.linspace(0.05, 0.99, 30)
    
    max_rel_err = 0.0
    
    for Om0, w0, wa in test_models:
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_values)
        if f_ref is None:
            continue
        
        for i, a in enumerate(a_values):
            if a <= 0:
                continue
                
            Om_a = _compute_om_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                return 10000.0
            
            f_approx = Om_a ** gamma
            f_true = f_ref[i]
            
            rel_err_f = abs(f_approx - f_true) / max(abs(f_true), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
            
            D_approx = _approx_D_ratio(a, Om0, w0, wa, coeffs)
            if D_approx < 0:
                return 10000.0
            D_true = D_ref[i]
            
            rel_err_D = abs(D_approx - D_true) / max(abs(D_true), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_D)
    
    return max_rel_err * 100

def construct():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    start_time = time.time()
    deadline = start_time + 85
    
    best_x = x0
    best_err = _growth_index_error(x0)
    
    for i in range(20):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(_growth_index_error, best_x, method='Nelder-Mead',
                             options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except Exception:
            break
    
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
# EVOLVE-BLOCK-END
