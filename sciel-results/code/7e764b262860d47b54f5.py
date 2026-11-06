"""Improved cosmology growth-index optimization using physics-informed Taylor matching and multi-start Nelder-Mead."""
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
        (0.22, -1.0, 0.0),
        (0.33, -1.0, 0.0),
        (0.28, -1.0, 0.2),
        (0.30, -1.05, 0.1),
    ]
    return models

def _precompute_references(models, a_values, num_steps=5000):
    """
    Pre-compute true f(a) and D(a)/D(1) for all models and a values
    by integrating the Riccati ODE for f on a fine log-a grid.
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
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    if not np.all(np.isfinite(coeffs)):
        return 1e10
        
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
        
        if not np.all(np.isfinite(gamma)):
            return 1e10
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            f_app = Om_m ** gamma
            
        if not np.all(np.isfinite(f_app)):
            return 1e10
        
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
        
        with np.errstate(over='ignore'):
            D_ratio = np.exp(integral_delta)
            
        if not np.all(np.isfinite(D_ratio)):
            return 1e10
            
        err_D = np.abs(D_ratio - 1.0)
        
        err = np.maximum(err_f, err_D)
        if not np.all(np.isfinite(err)):
            return 1e10
            
        local_max = np.max(err)
        if not np.isfinite(local_max):
            return 1e10
        if local_max > max_err:
            max_err = local_max
    
    return 100.0 * max_err

def _taylor_match_initial_guess(refs):
    """
    Physics-informed initial guess: match gamma and its derivatives
    at present epoch (a=1) and a high-redshift characteristic point.
    
    At a=1: Om_m = Om0, q = 1-Om0, u = 1+w0
    We want gamma(a=1) to match the effective gamma from the ODE solution.
    Also match the slope dgamma/dq and dgamma/du at characteristic points.
    """
    # Collect data for least-squares fit with Taylor-matching constraints
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_gamma_true = []
    
    for ref in refs:
        Om_m = ref['Om_m']
        q = ref['q']
        u = ref['u']
        wa = ref['wa']
        a_vals = ref['a']
        f_true = ref['f_true']
        
        valid = (Om_m > 0.01) & (Om_m < 0.99) & (f_true > 0.01)
        if np.sum(valid) < 5:
            continue
        
        gamma_true = np.log(f_true[valid]) / np.log(Om_m[valid])
        
        all_q.append(q[valid])
        all_u.append(u[valid])
        all_wa_a.append(wa * a_vals[valid])
        all_uq.append(u[valid] * q[valid])
        all_gamma_true.append(gamma_true)
    
    if not all_q:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    q_all = np.concatenate(all_q)
    u_all = np.concatenate(all_u)
    wa_a_all = np.concatenate(all_wa_a)
    uq_all = np.concatenate(all_uq)
    gamma_all = np.concatenate(all_gamma_true)
    
    # Build design matrix with additional Taylor-matching terms
    # The ansatz is gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # At a=1: gamma = c0 + c1*(1-Om0) + c2*(1-Om0)^2 + c3*(1+w0) + c4*wa + c5*(1+w0)*(1-Om0)
    # We want to weight points near a=1 more heavily to capture the present-epoch behavior
    
    A = np.column_stack([
        np.ones_like(q_all),
        q_all,
        q_all**2,
        u_all,
        wa_a_all,
        uq_all
    ])
    
    # Weight points: higher weight for a closer to 1 (present epoch)
    # Use a_values from the first ref as representative
    a_vals_ref = refs[0]['a']
    weights = 1.0 + 5.0 * np.ones_like(a_vals_ref)  # base weight
    # Add extra weight for points near a=1
    for i, a in enumerate(a_vals_ref):
        if a > 0.8:
            weights[i] += 3.0
        elif a > 0.5:
            weights[i] += 1.0
    
    # Extend weights to match all data points (repeat pattern)
    n_per_model = len(a_vals_ref)
    n_models = len(refs)
    full_weights = np.tile(weights, n_models)
    
    # Weighted least squares
    try:
        W = np.diag(full_weights)
        WA = W @ A
        Wb = W @ gamma_all
        coeffs, _, _, _ = np.linalg.lstsq(WA, Wb, rcond=None)
        coeffs = np.clip(coeffs, -5.0, 5.0)
        return coeffs
    except Exception:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

def _multi_start_optimize(refs, start_points, deadline, start_time):
    """Run Nelder-Mead from multiple starting points, returning best result."""
    best_score = np.inf
    best_params = None
    
    for sp in start_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 12.0:
            break
        
        try:
            res = minimize(
                _error_from_coeffs,
                sp,
                args=(refs,),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 250, 'adaptive': True}
            )
            if np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    safe_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        models = _cosmology_setup()
        a_values = np.linspace(0.1, 1.0, 60)
        
        refs = _precompute_references(models, a_values, num_steps=5000)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            return [float(x) for x in safe_coeffs]
        
        # Get initial guess via Taylor-matching weighted least squares
        x0 = _taylor_match_initial_guess(refs)
        if not np.all(np.isfinite(x0)):
            x0 = safe_coeffs.copy()
        x0 = np.clip(x0, -5.0, 5.0)
        
        # Evaluate seed and initial guess
        seed = safe_coeffs.copy()
        seed_score = _error_from_coeffs(seed, refs)
        x0_score = _error_from_coeffs(x0, refs)
        
        if x0_score < seed_score:
            best_score = x0_score
            best_params = x0.copy()
        else:
            best_score = seed_score
            best_params = seed.copy()
        
        # Build diverse start points
        start_points = [x0, seed]
        
        rng = np.random.default_rng(42)
        for _ in range(6):
            perturbation = rng.normal(0, 0.15, 6)
            sp = best_params + perturbation
            if np.all(np.isfinite(sp)) and np.all(np.abs(sp) < 5.0):
                start_points.append(sp)
        
        # Also try starts with specific physics-motivated offsets
        # Offset in c1 (q sensitivity)
        sp_c1 = best_params.copy()
        sp_c1[1] += 0.1
        start_points.append(sp_c1)
        sp_c1 = best_params.copy()
        sp_c1[1] -= 0.1
        start_points.append(sp_c1)
        
        # Offset in c3 (u sensitivity)
        sp_c3 = best_params.copy()
        sp_c3[3] += 0.05
        start_points.append(sp_c3)
        
        # Run multi-start optimization
        ms_score, ms_params = _multi_start_optimize(refs, start_points, deadline, start_time)
        if ms_params is not None and ms_score < best_score:
            best_score = ms_score
            best_params = ms_params.copy()
        
        # Polish pass
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            try:
                res2 = minimize(
                    _error_from_coeffs,
                    best_params,
                    args=(refs,),
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 200, 'adaptive': True}
                )
                if np.all(np.isfinite(res2.x)) and np.isfinite(res2.fun):
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_params = res2.x.copy()
            except Exception:
                pass
        
        # Final polish if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 5.0:
            try:
                res3 = minimize(
                    _error_from_coeffs,
                    best_params,
                    args=(refs,),
                    method='Nelder-Mead',
                    options={'xatol': 1e-9, 'fatol': 1e-10, 'maxiter': 150, 'adaptive': True}
                )
                if np.all(np.isfinite(res3.x)) and np.isfinite(res3.fun):
                    if res3.fun < best_score:
                        best_score = res3.fun
                        best_params = res3.x.copy()
            except Exception:
                pass
        
        best_params = np.clip(best_params, -10.0, 10.0)
        if not np.all(np.isfinite(best_params)):
            return [float(x) for x in safe_coeffs]
            
        return [float(x) for x in best_params]
    
    except Exception:
        return [float(x) for x in safe_coeffs]

# EVOLVE-BLOCK-END
