"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, simpson
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # 8-second safety margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.where(a > 1e-12, a, 1e-12)
    # E^2 = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa)) * exp(-3*wa*(1-a))
    E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1 + w0 + wa)) * np.exp(-3 * wa * (1 - a_safe))
    Om = Om0 * a_safe**(-3) / E2
    # Handle a <= 0
    if np.ndim(a) == 0:
        return Om if a > 0 else Om0
    return np.where(a > 0, Om, Om0)

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1 - Om
    u = 1 + w0 + wa * (1 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    Om_safe = np.maximum(Om, 1e-300)
    return Om_safe ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa, npts=300):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_arr = np.linspace(np.log(ai), 0.0, npts)
            a_arr = np.exp(x_arr)
            f_arr = _f_of_a(a_arr, coeffs, Om0, w0, wa)
            integral = simpson(f_arr, x=x_arr)
            D[i] = np.exp(-integral)
    return D

def _ode_reference_f(a, Om0, w0, wa, npts=1000):
    """Solve the GR ODE for f(a) and return f at given a values."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]) if np.any(a > 0) else 0.01, 0.005)
    x_max = 0.0
    x_min = np.log(a_min)
    
    # Initial condition at x=0 (a=1)
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5 * Om_at_1 + 0.5 * (1 - Om_at_1) * (1 + w0 + wa)
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5 * (3 * Om - 2) * f + 0.5 * 3 * Om
        return [dfdx]
    
    x_grid = np.linspace(x_min, x_max, npts)
    sol = solve_ivp(ode, [x_min, x_max], [f0], t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-11)
    
    f_ref = np.zeros(len(a))
    for i, ai in enumerate(a):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            # Use interpolation from solution
            f_ref[i] = np.interp(x_val, x_grid, sol.y[0])
    return f_ref

def _ode_reference_D(a, Om0, w0, wa, npts=1000):
    """Solve the GR ODE for D(a) and return normalized D(a)/D(1)."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]) if np.any(a > 0) else 0.01, 0.005)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    
    # Get f on the grid
    a_grid = np.exp(x_grid)
    f_grid = _ode_reference_f(a_grid, Om0, w0, wa, npts)
    
    # Cumulative integral
    from scipy.integrate import cumulative_trapezoid
    integral_from_min = cumulative_trapezoid(f_grid, x_grid, initial=0.0)
    total = integral_from_min[-1]
    
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            int_at_x = np.interp(x_val, x_grid, integral_from_min)
            int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

def _compute_error(coeffs, Om0, w0, wa, a_vals):
    """Compute max relative error for f and D."""
    f_approx = _f_of_a(a_vals, coeffs, Om0, w0, wa)
    f_ref = _ode_reference_f(a_vals, Om0, w0, wa)
    
    D_approx = _D_normalized(a_vals, coeffs, Om0, w0, wa)
    D_ref = _ode_reference_D(a_vals, Om0, w0, wa)
    
    mask_f = np.abs(f_ref) > 1e-6
    mask_D = np.abs(D_ref) > 1e-6
    
    err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])) if np.any(mask_f) else 0.0
    err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])) if np.any(mask_D) else 0.0
    
    return max(err_f, err_D)

# Validation cosmologies (representative CPL dark energy models)
_VALIDATION_MODELS = [
    (0.30, 0.0, 0.0),    # LambdaCDM
    (0.30, -0.5, 0.0),   # w = -0.5
    (0.30, -1.0, 0.5),   # CPL with wa=0.5
    (0.30, -1.0, -0.5),  # CPL with wa=-0.5
    (0.25, -0.9, 0.3),   # Different Om0
    (0.35, -1.1, 0.2),   # Another combination
]

_A_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

def _objective(coeffs):
    """Objective function: max error across all validation models."""
    if time.time() > _DEADLINE:
        return 1e6
    max_err = 0.0
    for Om0, w0, wa in _VALIDATION_MODELS:
        err = _compute_error(coeffs, Om0, w0, wa, _A_VALUES)
        max_err = max(max_err, err)
    return max_err

# Pre-computed best coefficients from optimization
_BEST_COEFFS = None

def _optimize():
    """Run optimization with time budget."""
    global _BEST_COEFFS
    t0 = time.time()
    
    # Initial guess: constant gamma seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Strategy: Boundary-matching asymptotic constraints
    # At small a (matter-dominated), gamma -> 6/11 ≈ 0.545
    # At large a with wa != 0, gamma behavior depends on w0, wa
    
    # Use Nelder-Mead first
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
    if time.time() < _DEADLINE and result.fun < best_val:
        best_val = result.fun
        best_x = result.x
    
    # Try L-BFGS-B with bounds
    bounds = [(-10, 10)] * 6
    if time.time() < _DEADLINE - 5:
        result2 = minimize(_objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    
    # Local refinement passes
    for _ in range(5):
        if time.time() >= _DEADLINE - 3:
            break
        result3 = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-11,
                                   'initial_simplex': 0.005 * np.ones(6)})
        if result3.fun < best_val:
            best_val = result3.fun
            best_x = result3.x
        else:
            break
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 2:
        result4 = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'initial_simplex': 0.001 * np.ones(6)})
        if result4.fun < best_val:
            best_val = result4.fun
            best_x = result4.x
    
    # Clamp to valid range
    best_x = np.clip(best_x, -10, 10)
    
    _BEST_COEFFS = list(best_x)
    return _BEST_COEFFS

def construct():
    # Return optimized coefficients
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    else:
        # Fallback: run optimization
        try:
            return _optimize()
        except Exception:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Run optimization at import time if possible
try:
    _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
