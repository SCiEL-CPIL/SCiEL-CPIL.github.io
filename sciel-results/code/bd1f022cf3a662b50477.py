"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # safety margin before 90s kill

# Validation cosmologies from the task description
# (Om0, w0, wa)
_VALIDATION_MODELS = [
    (0.30, 0.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
    (0.35, -1.1, 0.2),
]

# Dense grid of scale factors for evaluation
# We need to cover the range where the error is likely to be worst.
# a in [0.1, 1.0] is a good range.
_A_GRID = np.linspace(0.1, 1.0, 50)

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
    # Clip gamma to valid range to avoid numerical issues, 
    # but the objective function will penalize out-of-range gamma
    gamma_safe = np.clip(gamma, 0.1, 1.2)
    Om_safe = np.maximum(Om, 1e-300)
    return np.power(Om_safe, gamma_safe)

def _D_normalized_from_f(f_vals, a_grid):
    """
    Compute D(a)/D(1) given f(a) values on a_grid.
    D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    Let x = ln(a). Then integral is from x to 0 of f(e^x) dx.
    """
    x_grid = np.log(a_grid)
    # We need integral from x to 0.
    # Let I(x) = integral from x to 0 of f(e^t) dt.
    # I(0) = 0.
    # I(x) = - integral from 0 to x of f(e^t) dt.
    # So D(a)/D(1) = exp(I(ln(a))).
    
    # Cumulative integral from 0 to x
    # Using trapezoidal rule
    dx = np.diff(x_grid)
    integrand = f_vals
    cum_int = np.zeros(len(x_grid))
    cum_int[0] = 0.0
    for i in range(1, len(x_grid)):
        cum_int[i] = cum_int[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dx[i-1]
    
    # I(x) = - cum_int
    I_vals = -cum_int
    return np.exp(I_vals)

def _compute_reference(Om0, w0, wa):
    """
    Compute reference f and D for a given cosmology on _A_GRID.
    """
    a_grid = _A_GRID
    x_grid = np.log(a_grid)
    
    # Solve ODE for f
    # f' = -f^2 - 0.5*(3*Om - 2)*f + 1.5*Om
    # where ' is d/d ln a
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5*(3.0*Om - 2.0)*f + 1.5*Om
        return [dfdx]
    
    # Initial condition at a=1 (x=0)
    Om1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5 * Om1 + 0.5 * (1.0 - Om1) * (1.0 + w0 + wa)
    
    # Solve backwards from 0 to x_min
    x_min = x_grid[0]
    sol = solve_ivp(ode, [0.0, x_min], [f0], t_eval=x_grid[::-1], method='RK45', rtol=1e-9, atol=1e-11)
    
    f_ref = sol.y[0][::-1]
    
    # Compute D_ref
    D_ref = _D_normalized_from_f(f_ref, a_grid)
    
    return f_ref, D_ref

# Pre-compute reference data for all validation models
_REF_DATA = {}
for model in _VALIDATION_MODELS:
    Om0, w0, wa = model
    try:
        f_ref, D_ref = _compute_reference(Om0, w0, wa)
        _REF_DATA[model] = (f_ref, D_ref)
    except Exception as e:
        print(f"Error computing reference for {model}: {e}")
        # Fallback: use constant gamma approx for reference? No, better to skip or use simple approx
        # For now, skip if error occurs, but this shouldn't happen.

def _objective(coeffs):
    """
    Compute the max relative error across all validation models.
    """
    if time.time() > _DEADLINE:
        return 1e6
    
    max_err = 0.0
    c0, c1, c2, c3, c4, c5 = coeffs
    
    for model in _VALIDATION_MODELS:
        if model not in _REF_DATA:
            continue
            
        Om0, w0, wa = model
        f_ref, D_ref = _REF_DATA[model]
        
        # Compute approximation
        f_approx = _f_of_a(_A_GRID, coeffs, Om0, w0, wa)
        D_approx = _D_normalized_from_f(f_approx, _A_GRID)
        
        # Check gamma constraint
        gamma_vals = _gamma_of_a(_A_GRID, coeffs, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            # Penalty for violating gamma constraint
            # The more it violates, the higher the penalty
            violation = np.max(np.abs(np.clip(gamma_vals, 0.1, 1.2) - gamma_vals))
            return 1e6 + violation
        
        # Calculate relative errors
        # Mask out points where reference is too small to avoid division by zero issues
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

def _optimize():
    """
    Use Differential Evolution to find optimal coefficients.
    """
    # Bounds for coefficients
    # c0: around 0.5-0.6
    # c1, c2: small
    # c3: related to u, might be small
    # c4, c5: small
    bounds = [
        (0.3, 0.8),   # c0
        (-1.0, 1.0),  # c1
        (-1.0, 1.0),  # c2
        (-1.0, 1.0),  # c3
        (-1.0, 1.0),  # c4
        (-1.0, 1.0),  # c5
    ]
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            polish=True,
            update_interval=10,
            callback=None
        )
        
        # Check if result is valid
        if result.fun < 1e5: # If it didn't hit the penalty
            return list(result.x)
        else:
            # Fallback to seed if optimization failed or hit penalty
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            
    except Exception as e:
        print(f"Optimization error: {e}")
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

_BEST_COEFFS = None

def construct():
    """Return optimized 6 coefficients."""
    global _BEST_COEFFS
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    
    if time.time() > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    _BEST_COEFFS = _optimize()
    return _BEST_COEFFS

# Run optimization at import time
try:
    _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
