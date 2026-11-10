"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, cumulative_trapezoid

# Global deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # 5s margin

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if np.isscalar(a):
        if a <= 0:
            return 1.0
        de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
        om_m = Om0 * a**(-3.0)
        om_de = (1.0 - Om0) * de_factor
        return om_m / (om_m + om_de)
    a = np.asarray(a, dtype=float)
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a grid of cosmologies and a-values.
    Returns a list of tuples: (Om0, w0, wa, a_vals, f_refs, D_refs)
    """
    # Representative cosmologies covering a broad range of CPL parameters
    cosmos = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.30, -0.9, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.8, 0.3),
        (0.35, -1.2, 0.2),
        (0.28, -0.95, -0.2),
        (0.32, -1.05, 0.4),
        (0.26, -1.1, 0.1),
        (0.33, -0.9, -0.3),
        (0.31, -1.0, 0.3),
        (0.29, -1.1, -0.1),
    ]
    
    # a values: log spaced to capture early and late times
    a_vals = np.geomspace(0.05, 1.0, 60)
    
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        a_min = a_vals[0]
        
        def ode(t, y):
            a = np.exp(t)
            om_a = _Om_of_a(a, Om0, w0, wa)
            f_val = y[0]
            D_val = y[1]
            df_dlna = f_val - f_val**2 - 1.5 * om_a
            dD_dlna = -f_val * D_val
            return [df_dlna, dD_dlna]
        
        f_init = Om0 ** 0.55
        D_init = 1.0
        
        t_end = np.log(a_min)
        
        sol = solve_ivp(
            ode, [0.0, t_end], [f_init, D_init],
            method='DOP853',
            rtol=1e-10,
            atol=1e-13,
            dense_output=True
        )
        
        if not sol.success:
            continue
        
        # Evaluate at a_vals
        t_vals = np.log(a_vals)
        y_all = sol.sol(t_vals)
        f_refs = y_all[0]
        D_refs = y_all[1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
        
    return ref_data

def _error_function(coeffs, ref_data):
    """
    Compute max relative error for f and D across all reference data.
    coeffs: [c0, c1, c2, c3, c4, c5] (but c1=c2=0 in our constrained strategy)
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = om_a_vals ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_f))
        
        x = np.log(a_vals)
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_D))
        
    return max_err


def _smooth_objective(coeffs, ref_data, tau=0.01):
    """
    Smooth surrogate for the max-error objective using log-sum-exp.
    This provides gradient information for the optimizer.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    all_errors = []
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = om_a_vals ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            all_errors.append(rel_err_f)
        
        x = np.log(a_vals)
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            all_errors.append(rel_err_D)
    
    all_errs = np.concatenate(all_errors)
    
    # Log-sum-exp for smooth max approximation
    max_val = np.max(all_errs)
    if max_val > 1e8:
        return max_val
    lse = max_val + np.log(np.sum(np.exp((all_errs - max_val) / tau))) * tau
    return lse

def _optimize():
    # Pre-compute reference data
    if time.time() > DEADLINE - 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    
    if len(ref_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Physics-constrained affine strategy:
    # Fix c1=0 and c2=0 to enforce standard q-dependence structure.
    # Optimize only c0, c3, c4, c5.
    # Full 6-coeff vector: [c0, 0, 0, c3, c4, c5]
    
    # Reduced 4-parameter optimization
    def _reduced_error(x4):
        c0, c3, c4, c5 = x4
        return _error_function([c0, 0.0, 0.0, c3, c4, c5], ref_data)
    
    def _reduced_smooth(x4, tau=0.01):
        c0, c3, c4, c5 = x4
        return _smooth_objective([c0, 0.0, 0.0, c3, c4, c5], ref_data, tau)
    
    # Initial guess: standard gamma ~ 0.55 + small corrections
    x0_4 = [0.55, 0.0, 0.0, 0.0]
    
    # Bounds for the 4 free parameters
    bounds_4 = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    best_x4 = x0_4
    best_err = _reduced_error(best_x4)
    
    if time.time() > DEADLINE - 15:
        return [best_x4[0], 0.0, 0.0, best_x4[1], best_x4[2], best_x4[3]]
    
    # Phase 1: Global search with differential evolution on the reduced space
    try:
        def de_callback(xk, convergence=None):
            return time.time() > DEADLINE - 12
        
        de_result = differential_evolution(
            _reduced_error, bounds_4,
            maxiter=40,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='sobol',
            callback=de_callback,
            polish=False
        )
        if de_result.fun < best_err:
            best_err = de_result.fun
            best_x4 = de_result.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        return [best_x4[0], 0.0, 0.0, best_x4[1], best_x4[2], best_x4[3]]
    
    # Phase 2: Smooth surrogate polish (log-sum-exp provides smoother landscape)
    try:
        res_smooth = minimize(
            _reduced_smooth, best_x4, args=(0.005,),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 2000}
        )
        if res_smooth.fun < 1e8:
            err_at_res = _reduced_error(res_smooth.x)
            if err_at_res < best_err:
                best_err = err_at_res
                best_x4 = res_smooth.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 5:
        return [best_x4[0], 0.0, 0.0, best_x4[1], best_x4[2], best_x4[3]]
    
    # Phase 3: High-precision polish with exact max-error objective
    try:
        res = minimize(
            _reduced_error, best_x4,
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 3000}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_x4 = res.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 2:
        return [best_x4[0], 0.0, 0.0, best_x4[1], best_x4[2], best_x4[3]]
    
    # Phase 4: Final ultra-precise polish
    try:
        res2 = minimize(
            _reduced_error, best_x4,
            method='Nelder-Mead',
            options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 500, 'maxfev': 1500}
        )
        if res2.fun < best_err:
            best_err = res2.fun
            best_x4 = res2.x
    except Exception:
        pass
    
    # Clamp to valid range
    best_x4 = np.clip(best_x4, -10, 10)
    
    return [best_x4[0], 0.0, 0.0, best_x4[1], best_x4[2], best_x4[3]]

# Run optimization at module load
_t0 = time.time()
try:
    _coeffs = _optimize().tolist()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Verify coefficients are valid
if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
