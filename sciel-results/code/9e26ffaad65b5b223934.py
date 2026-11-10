"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, simpson
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # safety margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.where(a > 1e-10, a, 1e-10)
    de_factor = np.power(a_safe, 3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a_safe))
    Om = Om0 / (Om0 / np.power(a_safe, 3.0) + (1.0 - Om0) * de_factor)
    return Om

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    Om_safe = np.maximum(Om, 1e-300)
    return np.power(Om_safe, gamma)

def _D_normalized(a, coeffs, Om0, w0, wa, npts=200):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    a_min = max(np.min(a[a > 0]), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    a_grid = np.exp(x_grid)
    f_grid = _f_of_a(a_grid, coeffs, Om0, w0, wa)
    # Cumulative integral from x_min to each point
    dx = x_grid[1] - x_grid[0]
    cum_int = np.zeros(npts)
    for i in range(1, npts):
        cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * dx
    total = cum_int[-1]
    for i, ai in enumerate(a):
        if ai >= 1.0 - 1e-12:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            int_at_x = np.interp(x_val, x_grid, cum_int)
            int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

def _ode_reference_f(a, Om0, w0, wa, npts=300):
    """Solve the GR ODE for f(a) = d ln D / d ln a."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]), 0.01)
    x_min = np.log(a_min)
    
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5 * Om_at_1 + 0.5 * (1.0 - Om_at_1) * (1.0 + w0 + wa)
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5*(3.0*Om - 2.0)*f + 1.5*Om
        return [dfdx]
    
    x_grid = np.linspace(x_min, 0.0, npts)
    sol = solve_ivp(ode, [x_min, 0.0], [f0], t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-11)
    
    f_ref = np.zeros(len(a))
    for i, ai in enumerate(a):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            f_ref[i] = np.interp(x_val, x_grid, sol.y[0])
    return f_ref

def _ode_reference_D(a, Om0, w0, wa, npts=300):
    """Compute normalized growth factor from ODE reference f."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    a_grid = np.exp(x_grid)
    f_grid = _ode_reference_f(a_grid, Om0, w0, wa, npts)
    
    dx = x_grid[1] - x_grid[0]
    cum_int = np.zeros(npts)
    for i in range(1, npts):
        cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * dx
    total = cum_int[-1]
    
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            int_at_x = np.interp(x_val, x_grid, cum_int)
            int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

# Validation cosmologies
_VALIDATION_MODELS = [
    (0.30, 0.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
    (0.35, -1.1, 0.2),
]

# Smaller set for optimization (faster)
_OPT_MODELS = [
    (0.30, 0.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
]

_A_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

_REF_CACHE = {}

def _get_reference(Om0, w0, wa):
    key = (Om0, w0, wa)
    if key not in _REF_CACHE:
        f_ref = _ode_reference_f(_A_VALUES, Om0, w0, wa)
        D_ref = _ode_reference_D(_A_VALUES, Om0, w0, wa)
        _REF_CACHE[key] = (f_ref, D_ref)
    return _REF_CACHE[key]

def _objective_full(coeffs):
    """Max relative error across all validation models."""
    if time.time() > _DEADLINE:
        return 1e6
    
    max_err = 0.0
    for Om0, w0, wa in _VALIDATION_MODELS:
        f_ref, D_ref = _get_reference(Om0, w0, wa)
        
        f_approx = _f_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        D_approx = _D_normalized(_A_VALUES, coeffs, Om0, w0, wa)
        
        gamma_vals = _gamma_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e6
        
        mask_f = np.abs(f_ref) > 1e-6
        mask_D = np.abs(D_ref) > 1e-6
        
        if np.any(mask_f):
            err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f]))
        else:
            err_f = 0.0
        
        if np.any(mask_D):
            err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D]))
        else:
            err_D = 0.0
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

def _objective_fast(coeffs):
    """Max relative error across optimization subset (faster)."""
    if time.time() > _DEADLINE:
        return 1e6
    
    max_err = 0.0
    for Om0, w0, wa in _OPT_MODELS:
        f_ref, D_ref = _get_reference(Om0, w0, wa)
        
        f_approx = _f_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        D_approx = _D_normalized(_A_VALUES, coeffs, Om0, w0, wa)
        
        gamma_vals = _gamma_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e6
        
        mask_f = np.abs(f_ref) > 1e-6
        mask_D = np.abs(D_ref) > 1e-6
        
        if np.any(mask_f):
            err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f]))
        else:
            err_f = 0.0
        
        if np.any(mask_D):
            err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D]))
        else:
            err_D = 0.0
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

def _objective_smooth(coeffs):
    """Smoothed objective: weighted combo of max-error and mean-squared error for DE."""
    if time.time() > _DEADLINE:
        return 1e6
    
    max_err = 0.0
    mse = 0.0
    count = 0
    for Om0, w0, wa in _OPT_MODELS:
        f_ref, D_ref = _get_reference(Om0, w0, wa)
        
        f_approx = _f_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        D_approx = _D_normalized(_A_VALUES, coeffs, Om0, w0, wa)
        
        gamma_vals = _gamma_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e6
        
        mask_f = np.abs(f_ref) > 1e-6
        mask_D = np.abs(D_ref) > 1e-6
        
        if np.any(mask_f):
            rel_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
            max_err = max(max_err, np.max(rel_f))
            mse += np.mean(rel_f**2)
            count += len(rel_f)
        
        if np.any(mask_D):
            rel_D = np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])
            max_err = max(max_err, np.max(rel_D))
            mse += np.mean(rel_D**2)
            count += len(rel_D)
    
    if count == 0:
        return 1e6
    
    mse /= count
    # Blend: emphasize max error but smooth with MSE
    return 0.7 * max_err + 0.3 * mse

_BEST_COEFFS = None

def _optimize():
    """Run optimization with time budget using DE + local refinement."""
    global _BEST_COEFFS
    t0 = time.time()
    deadline = t0 + 75.0
    
    # Pre-compute references for all models
    for Om0, w0, wa in _VALIDATION_MODELS:
        _get_reference(Om0, w0, wa)
    
    best_coeffs = None
    best_val = 1e6
    
    # Phase 1: Differential Evolution (global search) with smooth objective
    if time.time() < deadline - 30:
        bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0),
                  (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
        
        def _de_wrapper(x):
            if time.time() > deadline:
                return 1e6
            return _objective_smooth(x)
        
        try:
            de_result = differential_evolution(
                _de_wrapper, bounds, seed=42, maxiter=100,
                popsize=15, tol=1e-8, mutation=(0.3, 1.0),
                recombination=0.7, polish=False,
                init='sobol', update_interval=10
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead on fast objective
    if best_coeffs is None:
        best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_val = _objective_fast(best_coeffs)
    
    if time.time() < deadline - 20:
        result = minimize(_objective_fast, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
    
    # Phase 3: L-BFGS-B polish
    if time.time() < deadline - 15:
        bounds = [(-10, 10)] * 6
        result2 = minimize(_objective_fast, best_coeffs, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14})
        if result2.fun < best_val:
            best_val = result2.fun
            best_coeffs = result2.x
    
    # Phase 4: Multi-start local refinement from perturbed best
    np.random.seed(123)
    for start in range(5):
        if time.time() >= deadline - 10:
            break
        x0 = best_coeffs + 0.05 * np.random.randn(6)
        x0 = np.clip(x0, -10, 10)
        result3 = minimize(_objective_fast, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if result3.fun < best_val:
            best_val = result3.fun
            best_coeffs = result3.x
    
    # Phase 5: Final polish with full validation objective
    if time.time() < deadline - 5:
        result4 = minimize(_objective_full, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13,
                                   'initial_simplex': 0.005 * np.ones(6)})
        if result4.fun < best_val:
            best_val = result4.fun
            best_coeffs = result4.x
    
    # Phase 6: L-BFGS-B final polish on full objective
    if time.time() < deadline - 2:
        bounds = [(-10, 10)] * 6
        result5 = minimize(_objective_full, best_coeffs, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-15})
        if result5.fun < best_val:
            best_val = result5.fun
            best_coeffs = result5.x
    
    _BEST_COEFFS = list(np.clip(best_coeffs, -10, 10))
    return _BEST_COEFFS

def construct():
    """Return optimized 6 coefficients."""
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    else:
        return _optimize()

# Run optimization at import time
try:
    _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
