"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, least_squares
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
        (0.30, -1.0, 0.1),
        (0.26, -1.05, -0.2),
        (0.30, -1.0, 0.3),
        (0.25, -1.0, -0.4),
        (0.33, -0.85, 0.15),
        (0.22, -1.15, 0.45),
        (0.28, -0.95, -0.15),
    ]
    return models

def _precompute_references(models, a_values, num_steps=10000):
    """
    Pre-compute true f(a) and D(a)/D(1) for all models and a values
    using RK4 integration of the Riccati ODE for f on a fine log-a grid.
    """
    refs = []
    a_min = float(np.min(a_values))
    x_grid = np.linspace(0.0, np.log(a_min), num_steps + 1)
    a_grid = np.exp(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    x_targets = np.log(a_values)
    
    for (Om0, w0, wa) in models:
        Om_m_grid = _Om_m_of_a(a_grid, Om0, w0, wa)
        
        f_vals = np.zeros(num_steps + 1)
        lnD_vals = np.zeros(num_steps + 1)
        f_vals[0] = 0.5
        lnD_vals[0] = 0.0
        
        # RK4 integration
        for i in range(num_steps):
            f_curr = f_vals[i]
            om_i = Om_m_grid[i]
            om_half = Om_m_grid[min(i + 1, num_steps)]  # approximate at midpoint
            
            def dfdx(f, om):
                return -f * (2.0 + f) + 1.5 * om
            
            k1 = dfdx(f_curr, om_i)
            k2 = dfdx(f_curr + 0.5 * k1 * dx, om_i)
            k3 = dfdx(f_curr + 0.5 * k2 * dx, om_i)
            k4 = dfdx(f_curr + k3 * dx, om_i)
            f_vals[i + 1] = f_curr + (dx / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # For lnD, use trapezoidal on f
            lnD_vals[i + 1] = lnD_vals[i] + 0.5 * (f_curr + f_vals[i+1]) * dx
        
        f_true = np.interp(x_targets, x_grid, f_vals)
        lnD_true = np.interp(x_targets, x_grid, lnD_vals)
        D_true = np.exp(lnD_true)
        
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
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_app = Om_m ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = np.abs(f_app - f_true) / np.abs(f_true)
        err_f = np.where(np.isfinite(err_f), err_f, 1e10)
        
        x_vals = np.log(a_vals)
        delta_f = f_app - f_true
        
        dx_arr = np.diff(x_vals)
        trapz_vals = np.zeros(len(x_vals))
        for i in range(1, len(x_vals)):
            trapz_vals[i] = trapz_vals[i-1] + 0.5 * (delta_f[i-1] + delta_f[i]) * dx_arr[i-1]
        
        trapz_total = trapz_vals[-1]
        integral_delta = trapz_vals - trapz_total
        
        D_ratio = np.exp(integral_delta)
        err_D = np.abs(D_ratio - 1.0)
        
        err = np.maximum(err_f, err_D)
        local_max = np.max(err)
        if local_max > max_err:
            max_err = local_max
    
    return 100.0 * max_err

def _residuals_for_ls(coeffs, refs):
    """
    Return a flat vector of residuals for least_squares.
    Each residual is the relative error in f or D at each point.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    residuals = []
    
    for ref in refs:
        Om_m = ref['Om_m']
        q = ref['q']
        u = ref['u']
        wa = ref['wa']
        a_vals = ref['a']
        f_true = ref['f_true']
        D_true = ref['D_true']
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return np.ones(len(f_true) * 2) * 1e5
        
        f_app = Om_m ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = (f_app - f_true) / np.abs(f_true)
        err_f = np.where(np.isfinite(err_f), err_f, 1e5)
        
        x_vals = np.log(a_vals)
        delta_f = f_app - f_true
        
        dx_arr = np.diff(x_vals)
        trapz_vals = np.zeros(len(x_vals))
        for i in range(1, len(x_vals)):
            trapz_vals[i] = trapz_vals[i-1] + 0.5 * (delta_f[i-1] + delta_f[i]) * dx_arr[i-1]
        
        trapz_total = trapz_vals[-1]
        integral_delta = trapz_vals - trapz_total
        
        D_ratio = np.exp(integral_delta)
        err_D = D_ratio - 1.0
        
        residuals.append(err_f)
        residuals.append(err_D)
    
    return np.concatenate(residuals)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    models = _cosmology_setup()
    a_values = np.linspace(0.1, 1.0, 40)
    
    refs = _precompute_references(models, a_values, num_steps=10000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_params = x0.copy()
    best_score = _error_from_coeffs(x0, refs)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return [float(x) for x in best_params]
    
    # Phase 1: Differential Evolution (global) - structural change from local-only optimization
    from scipy.optimize import differential_evolution
    try:
        bounds_de = [(-10.0, 10.0)] * 6
        res_de = differential_evolution(
            _error_from_coeffs,
            bounds_de,
            args=(refs,),
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            mutation=(0.3, 1.2),
            recombination=0.7,
            polish=True,
            init='latinhypercube'
        )
        if res_de.success and res_de.fun < best_score:
            best_score = res_de.fun
            best_params = res_de.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15.0:
        best_params = np.clip(best_params, -10.0, 10.0)
        return [float(x) for x in best_params]
    
    # Phase 2: Local least-squares polish from DE result
    try:
        res_ls = least_squares(
            _residuals_for_ls,
            best_params,
            args=(refs,),
            bounds=(-10.0, 10.0),
            method='trf',
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300
        )
        candidate = np.clip(res_ls.x, -10.0, 10.0)
        score_candidate = _error_from_coeffs(candidate, refs)
        if score_candidate < best_score:
            best_score = score_candidate
            best_params = candidate
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        best_params = np.clip(best_params, -10.0, 10.0)
        return [float(x) for x in best_params]
    
    # Phase 3: Final Nelder-Mead fine polish
    try:
        res_nm = minimize(
            _error_from_coeffs,
            best_params,
            args=(refs,),
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 200, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x
    except Exception:
        pass
    
    best_params = np.clip(best_params, -10.0, 10.0)
    
    return [float(x) for x in best_params]

# EVOLVE-BLOCK-END
