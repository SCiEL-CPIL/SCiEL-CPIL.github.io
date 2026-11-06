"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def _H2_of_a(a, Om0, w0, wa):
    """Hubble parameter squared normalized to H0^2 for CPL dark energy."""
    Om_m = Om0
    Om_de = 1.0 - Om0
    if Om_de < 0:
        Om_de = 1e-10
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe ** (-3)
    exponent = -3.0 * (1.0 + w0 + wa)
    exp_factor = 3.0 * wa * (1.0 / a_safe - 1.0)
    term_de = Om_de * (a_safe ** exponent) * np.exp(exp_factor)
    return term_m + term_de

def _Om_m_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    H2 = _H2_of_a(a, Om0, w0, wa)
    return Om0 * a ** (-3) / H2

def _cosmology_setup():
    """Representative CPL models for local optimization."""
    models = [
        (0.30, -1.0, 0.0),
        (0.28, -1.1, 0.4),
        (0.32, -0.9, 0.1),
        (0.25, -1.2, 0.5),
        (0.30, -0.8, -0.5),
        (0.35, -1.0, 0.2),
        (0.20, -1.3, 0.6),
        (0.27, -0.95, -0.3),
        (0.30, -1.0, 0.5),
        (0.30, -1.2, 0.0),
        (0.25, -0.8, 0.0),
        (0.35, -1.1, 0.3),
    ]
    return models

def _precompute_references(models, a_values, num_steps=8000):
    """
    Pre-compute true f(a) and D(a)/D(1) for all models and a values
    by integrating the Riccati ODE for f on a fine log-a grid.
    """
    refs = []
    a_min = float(np.min(a_values))
    # Fine grid in x = ln(a) from 0 down to ln(a_min)
    x_grid = np.linspace(0.0, np.log(a_min), num_steps + 1)
    a_grid = np.exp(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    x_targets = np.log(a_values)
    
    for (Om0, w0, wa) in models:
        # Precompute Om_m on the fine grid
        Om_m_grid = _Om_m_of_a(a_grid, Om0, w0, wa)
        
        # Integrate ODE: df/dx = -f(2+f) + 1.5*Om_m(a)
        f_vals = np.zeros(num_steps + 1)
        lnD_vals = np.zeros(num_steps + 1)
        f_vals[0] = 0.5  # f(1) ~ 0.5
        lnD_vals[0] = 0.0  # D(1)/D(1) = 1
        
        # Use RK4 for better accuracy
        for i in range(num_steps):
            x_curr = x_grid[i]
            a_curr = a_grid[i]
            Om_m_curr = Om_m_grid[i]
            
            f_curr = f_vals[i]
            
            def df_dx(x, f):
                return -f * (2.0 + f) + 1.5 * Om_m_curr
            
            k1 = df_dx(x_curr, f_curr)
            k2 = df_dx(x_curr + 0.5 * dx, f_curr + 0.5 * dx * k1)
            k3 = df_dx(x_curr + 0.5 * dx, f_curr + 0.5 * dx * k2)
            k4 = df_dx(x_curr + dx, f_curr + dx * k3)
            
            f_vals[i + 1] = f_curr + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            lnD_vals[i + 1] = lnD_vals[i] + 0.5 * (f_vals[i] + f_vals[i + 1]) * dx
        
        # Interpolate to target a_values
        f_true = np.interp(x_targets, x_grid, f_vals)
        lnD_true = np.interp(x_targets, x_grid, lnD_vals)
        D_true = np.exp(lnD_true)
        
        # Precompute Om_m, q, u at target points
        Om_m_targets = _Om_m_of_a(a_values, Om0, w0, wa)
        q_targets = 1.0 - Om_m_targets
        u_targets = 1.0 + w0 + wa * (1.0 - a_values)
        
        refs.append({
            'f_true': f_true,
            'D_true': D_true,
            'Om_m': Om_m_targets,
            'q': q_targets,
            'u': u_targets,
            'w0': w0,
            'wa': wa,
            'a': a_values,
        })
    return refs

def _error_from_coeffs(coeffs, refs):
    """
    Compute 100 * max relative error of f and D over all refs.
    Uses precomputed references. Vectorized.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for ref in refs:
        Om_m = ref['Om_m']
        q = ref['q']
        u = ref['u']
        wa = ref['wa']
        a_vals = ref['a']
        f_true = ref['f_true']
        D_true = ref['D_true']
        
        # Compute gamma
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        # f approximation
        f_app = Om_m ** gamma
        
        # Error in f
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = np.abs(f_app - f_true) / np.abs(f_true)
        err_f = np.where(np.isfinite(err_f), err_f, 1e10)
        
        # For D: D_app(a) = exp(integral_1^a f_app(x) dx/x)
        # We need to integrate f_app over x from 0 to ln(a_i)
        # Since we have f_app at discrete a_values, use cumulative trapezoidal
        # But a_values may not be sorted in x. Let's compute on a fine grid.
        # Actually, for speed, approximate: if err_f is small, err_D ~ integral err_f
        # Better: compute D_app by integrating f_app on the same fine grid used for reference.
        # But that requires re-integrating. For speed, use a simpler approximation:
        # D_app / D_true = exp(integral (f_app - f_true) dx)
        # So err_D = |exp(integral delta_f dx) - 1|
        # We can estimate integral delta_f dx by trapezoidal on a_values.
        
        # Sort a_values for integration (they should already be sorted)
        x_vals = np.log(a_vals)
        # We need integral from 0 to x_i of delta_f dx
        # delta_f = f_app - f_true
        delta_f = f_app - f_true
        
        # Compute cumulative integral of delta_f using trapezoidal rule
        # x_vals is in [ln(a_min), 0], increasing
        # We want integral from 0 to x_i (x_i < 0)
        # = - integral from x_i to 0
        
        # Simpler: compute integral from x_min to x_i for each i
        # Use cumulative trapezoidal
        # x_vals is increasing (from ln(a_min) to 0)
        # dx = diff(x_vals)
        dx_arr = np.diff(x_vals)
        # Trapezoidal cumulative
        trapz_vals = np.zeros(len(x_vals))
        for i in range(1, len(x_vals)):
            trapz_vals[i] = trapz_vals[i-1] + 0.5 * (delta_f[i-1] + delta_f[i]) * dx_arr[i-1]
        
        # trapz_vals[i] = integral from x_vals[0] to x_vals[i] of delta_f dx
        # We want integral from 0 to x_vals[i] = - integral from x_vals[i] to 0
        # = -(trapz_total - trapz_vals[i]) where trapz_total = integral from x_vals[0] to 0
        trapz_total = trapz_vals[-1]
        # integral from x_vals[i] to 0 = trapz_total - trapz_vals[i]
        # integral from 0 to x_vals[i] = -(trapz_total - trapz_vals[i]) = trapz_vals[i] - trapz_total
        integral_delta = trapz_vals - trapz_total
        
        D_ratio = np.exp(integral_delta)
        err_D = np.abs(D_ratio - 1.0)
        
        err = np.maximum(err_f, err_D)
        local_max = np.max(err)
        if local_max > max_err:
            max_err = local_max
    
    return 100.0 * max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    models = _cosmology_setup()
    a_values = np.linspace(0.1, 1.0, 50)
    
    # Precompute references (this is the expensive part, done once)
    refs = _precompute_references(models, a_values, num_steps=8000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _error_from_coeffs(x0, refs)
    best_params = x0.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return [float(x) for x in best_params]
    
    # Nelder-Mead optimization
    try:
        res = minimize(
            _error_from_coeffs,
            x0,
            args=(refs,),
            method='Nelder-Mead',
            options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15.0:
        return [float(x) for x in best_params]
    
    # Polish pass with tighter tolerances
    try:
        res2 = minimize(
            _error_from_coeffs,
            best_params,
            args=(refs,),
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': 300, 'adaptive': True}
        )
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x
    except Exception:
        pass
    
    # Final polish with even tighter tolerances if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res3 = minimize(
                _error_from_coeffs,
                best_params,
                args=(refs,),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 200, 'adaptive': True}
            )
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x
        except Exception:
            pass
    
    # Clamp to [-10, 10]
    best_params = np.clip(best_params, -10.0, 10.0)
    
    return [float(x) for x in best_params]

# EVOLVE-BLOCK-END
