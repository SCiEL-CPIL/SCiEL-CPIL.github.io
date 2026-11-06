"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import time

# EVOLVE-BLOCK-START

# Validation cosmologies: (Om0, w0, wa)
_VAL_MODELS = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -0.9, -0.3),
    (0.30, -1.1, 0.5),
    (0.25, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.5, -0.5),
    (0.30, -1.5, 0.5),
    (0.20, -1.0, 0.0),
    (0.40, -1.0, 0.0),
    (0.30, -0.9, 0.0),
    (0.30, -1.0, 1.0),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
    (0.35, -1.1, 0.3),
    (0.30, -0.8, 0.2),
    (0.30, -1.2, -0.3),
    (0.28, -1.0, 0.0),
    (0.32, -1.0, 0.0),
]

# a values to check
_VAL_A = np.geomspace(0.05, 1.0, 50)

# Precompute reference solutions
_REF_CACHE = {}

def _Om(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    Om_de0 = 1.0 - Om0
    log_a = np.log(a)
    de_log = -3.0*(1.0+w0)*log_a - 3.0*wa*(a-1.0)
    de_factor = np.exp(de_log)
    E2 = Om0 * a**(-3.0) + Om_de0 * de_factor
    return Om0 * a**(-3.0) / E2

def _growth_ode(x, y, Om0, w0, wa):
    """Growth ODE: D'' + (3/2 + q) D' - (3/2) Om_m D = 0, x=ln(a)"""
    a = np.exp(x)
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    D, Dp = y
    Dpp = -(3.0/2.0 + q) * Dp + (3.0/2.0) * Om * D
    return [Dp, Dpp]

def _solve_growth_ref(Om0, w0, wa, a_vals):
    """Solve growth ODE and return normalized D(a)/D(1) and f(a)=D'/D."""
    key = (Om0, w0, wa)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    
    x0 = np.log(0.03)
    x1 = 0.0
    y0 = [0.03, 1.0]
    sol = solve_ivp(_growth_ode, [x0, x1], y0, args=(Om0, w0, wa),
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.005)
    if not sol.success:
        return None, None
    
    D_ref = np.zeros_like(a_vals)
    f_ref = np.zeros_like(a_vals)
    for i, a in enumerate(a_vals):
        x = np.log(a)
        y = sol.sol(x)
        D_ref[i] = y[0]
        f_ref[i] = y[1] / y[0]
    
    D_norm = D_ref / D_ref[-1]
    result = (D_norm, f_ref)
    _REF_CACHE[key] = result
    return result

def _compute_error_fast(coeffs):
    """Compute max relative error, vectorized per model."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for (Om0, w0, wa) in _VAL_MODELS:
        D_ref, f_ref = _solve_growth_ref(Om0, w0, wa, _VAL_A)
        if D_ref is None:
            continue
        
        a = _VAL_A
        Om = _Om(a, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_app = Om ** gamma
        
        mask = np.abs(f_ref) > 1e-10
        if np.any(mask):
            err_f = np.abs(f_app[mask] - f_ref[mask]) / np.abs(f_ref[mask])
            max_err = max(max_err, float(np.max(err_f)))
        
        x = np.log(a)
        dx = np.diff(x)
        f_avg = 0.5 * (f_app[:-1] + f_app[1:])
        cum_int = np.concatenate([[0.0], np.cumsum(f_avg * dx)])
        total_int = cum_int[-1]
        int_ai_to_1 = total_int - cum_int
        D_app = np.exp(-int_ai_to_1)
        
        mask_d = np.abs(D_ref) > 1e-10
        if np.any(mask_d):
            err_d = np.abs(D_app[mask_d] - D_ref[mask_d]) / np.abs(D_ref[mask_d])
            max_err = max(max_err, float(np.max(err_d)))
    
    return max_err

def _objective(coeffs):
    try:
        err = _compute_error_fast(coeffs)
        if err >= 1e10:
            return 1e10
        return 100.0 * err
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 72.0
    
    bounds = [(-10, 10)] * 6
    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Quick local optimization from seed
    try:
        res = minimize(_objective, seed, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        best_coeffs = np.array(res.x)
        best_obj = res.fun
    except Exception:
        best_coeffs = np.array(seed)
        best_obj = _objective(seed)
    
    if time.time() > deadline:
        return [float(x) for x in np.clip(best_coeffs, -10, 10)]
    
    # Differential Evolution
    try:
        de_result = differential_evolution(
            _objective, bounds,
            seed=42,
            maxiter=80,
            popsize=12,
            tol=1e-5,
            mutation=(0.4, 1.0),
            recombination=0.7,
            polish=True,
            init='latinhypercube'
        )
        if de_result.fun < best_obj:
            best_coeffs = de_result.x
            best_obj = de_result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return [float(x) for x in np.clip(best_coeffs, -10, 10)]
    
    # Final polish
    try:
        res2 = minimize(_objective, best_coeffs, method='Nelder-Mead',
                        options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < best_obj:
            best_coeffs = res2.x
            best_obj = res2.fun
    except Exception:
        pass
    
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return [float(x) for x in best_coeffs]

# EVOLVE-BLOCK-END
