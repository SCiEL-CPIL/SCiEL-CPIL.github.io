"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    de_factor = np.exp(-3 * ((1 + w0) * np.log(a) + wa * (a - 1)))
    om_m = Om0 * a**(-3)
    om_de = (1 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _f_reference(a, Om0, w0, wa):
    """Reference growth rate using a numerical ODE solve (fast version)."""
    if a >= 1.0:
        return Om0 ** 0.55
    
    # Use a coarse grid for speed
    n_steps = 50
    ln_a = np.linspace(0.0, np.log(a), n_steps + 1)
    f = Om0 ** 0.55
    
    for i in range(n_steps):
        a_i = np.exp(ln_a[i])
        om_i = _Om_of_a(a_i, Om0, w0, wa)
        df_dlna = f - f**2 - 1.5 * om_i
        f += df_dlna * (ln_a[i+1] - ln_a[i])
    
    return f

def _error_for_cosmology(coeffs, Om0, w0, wa, a_vals):
    """Compute max relative error for f and D for one cosmology."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    # Compute f and D for all a values
    f_approx_list = []
    f_ref_list = []
    
    for a in a_vals:
        om_a = _Om_of_a(a, Om0, w0, wa)
        if om_a <= 0 or om_a >= 1:
            continue
            
        q = 1 - om_a
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if gamma < 0.1 or gamma > 1.2:
            return 1e10
            
        f_approx = om_a ** gamma
        f_ref = _f_reference(a, Om0, w0, wa)
        
        if abs(f_ref) > 1e-10:
            rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
            max_err = max(max_err, rel_err_f)
        
        f_approx_list.append((a, f_approx))
        f_ref_list.append((a, f_ref))
    
    # Compute D(a)/D(1) using trapezoidal rule
    if len(f_approx_list) > 1:
        D_approx = 1.0
        D_ref = 1.0
        for i in range(1, len(f_approx_list)):
            a1, f1 = f_approx_list[i-1]
            a2, f2 = f_approx_list[i]
            a1r, f1r = f_ref_list[i-1]
            a2r, f2r = f_ref_list[i]
            
            if a1 > 0 and a2 > 0:
                da_ln = np.log(a2) - np.log(a1)
                D_approx *= np.exp(-(f1 + f2) / 2 * da_ln)
                D_ref *= np.exp(-(f1r + f2r) / 2 * da_ln)
        
        if abs(D_ref) > 1e-10:
            rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
            max_err = max(max_err, rel_err_D)
    
    return max_err

def _optimize():
    """Optimize coefficients with physics-constrained affine strategy."""
    # Fix c1=0 and c2=0 to enforce standard q-dependence
    # Optimize c0, c3, c4, c5
    test_cosmos = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.9, 0.3),
        (0.35, -1.1, 0.2),
    ]
    a_vals = np.linspace(0.05, 1.0, 30)
    
    def obj_4d(params):
        """Objective function for 4 free parameters (c0, c3, c4, c5)."""
        c0, c3, c4, c5 = params
        coeffs = [c0, 0.0, 0.0, c3, c4, c5]
        max_e = 0.0
        for Om0, w0, wa in test_cosmos:
            e = _error_for_cosmology(coeffs, Om0, w0, wa, a_vals)
            max_e = max(max_e, e)
        return max_e
    
    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0])
    
    # Bounds: c0 in [0.1, 1.2], others in [-10, 10]
    bounds = [(0.1, 1.2), (-10, 10), (-10, 10), (-10, 10)]
    
    # Use Nelder-Mead with tighter tolerance
    result = minimize(
        obj_4d, x0, method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'maxfev': 1000}
    )
    
    best_params = result.x
    best_err = result.fun
    
    # Try a few more random restarts if time allows
    deadline = time.time() + 30  # 30 seconds for additional search
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Random restart within reasonable bounds
        x_rand = np.array([
            np.random.uniform(0.3, 0.8),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2)
        ])
        
        res = minimize(
            obj_4d, x_rand, method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'maxfev': 500}
        )
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
    
    # Final polish with even tighter tolerance
    final_res = minimize(
        obj_4d, best_params, method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 300, 'maxfev': 600}
    )
    
    if final_res.fun < best_err:
        best_params = final_res.x
        best_err = final_res.fun
    
    # Convert back to 6 coefficients
    c0, c3, c4, c5 = best_params
    coeffs = [c0, 0.0, 0.0, c3, c4, c5]
    return np.clip(coeffs, -10, 10)

_t0 = time.time()
try:
    _coeffs = _optimize().tolist()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
