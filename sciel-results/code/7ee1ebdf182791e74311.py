"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    # E^2 = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa)) * exp(-3*wa*(1-a))
    # Om(a) = Om0 * a^-3 / E^2
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0):
        return np.ones_like(a) * Om0
    E2 = Om0 * a**(-3) + (1-Om0) * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
    return Om0 * a**(-3) / E2

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1 - Om
    u = 1 + w0 + wa * (1 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    # Clamp gamma to [0.1, 1.2]
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa, npts=200):
    """Compute D(a)/D(1) = exp(-integral from a to 1 of f(a')/a' da')."""
    # Using the formula: D(a)/D(1) = exp(-∫_a^1 f(a')/a' da')
    # Substitute x = ln(a'), dx = da'/a'
    # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            # Integrate from ln(ai) to 0
            x_arr = np.linspace(np.log(ai), 0.0, npts)
            a_arr = np.exp(x_arr)
            f_arr = _f_of_a(a_arr, coeffs, Om0, w0, wa)
            # Simpson's rule
            from scipy.integrate import simpson
            integral = simpson(f_arr, x=x_arr)
            D[i] = np.exp(-integral)
    return D

def _ode_reference_f(a, Om0, w0, wa, npts=500):
    """Solve the GR ODE for f(a) and return f at given a values."""
    # ODE: f' + f^2 + (1.5*Om - 1)/2 * f - (1.5*Om)/2 = 0 (in terms of ln a)
    # Let x = ln(a), d/dx = a*d/da
    # The growth equation in terms of x:
    # f' + f^2 + 0.5*(3*Om - 2)*f - 0.5*3*Om = 0
    # where f' = df/dx
    
    a = np.asarray(a, dtype=float)
    # We need to integrate from a=1 backwards
    # f(1) is determined by boundary condition
    # For a flat universe, f(1) ≈ 0.5*Om0 + 0.5*(1-Om0)*(1+w0+wa)
    # More precisely, use the asymptotic form
    
    # Integrate from x=0 (a=1) to x=ln(a_min)
    a_min = max(np.min(a), 0.01)
    x_max = 0.0
    x_min = np.log(a_min)
    
    # Initial condition at x=0: f(0)
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5*Om_at_1 + 0.5*(1-Om_at_1)*(1+w0+wa)
    
    # Define ODE: df/dx = -f^2 - 0.5*(3*Om - 2)*f + 0.5*3*Om
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5*(3*Om - 2)*f + 0.5*3*Om
        return [dfdx]
    
    x_eval = np.sort(a)
    x_eval = x_eval[x_eval > 0]
    if len(x_eval) == 0:
        return np.ones(len(a)) * f0
    
    x_grid = np.linspace(x_min, x_max, npts)
    sol = solve_ivp(ode, [x_min, x_max], [f0], t_eval=x_grid, method='RK45', rtol=1e-8, atol=1e-10)
    
    f_ref = np.zeros(len(a))
    for i, ai in enumerate(a):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            # Interpolate
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                f_ref[i] = sol.y[0, 0]
            elif idx >= len(x_grid):
                f_ref[i] = sol.y[0, -1]
            else:
                # Linear interpolation
                x0, x1 = x_grid[idx-1], x_grid[idx]
                f0v, f1v = sol.y[0, idx-1], sol.y[0, idx]
                t = (x_val - x0) / (x1 - x0)
                f_ref[i] = f0v + t * (f1v - f0v)
    return f_ref

def _ode_reference_D(a, Om0, w0, wa, npts=500):
    """Solve the GR ODE for D(a) and return normalized D(a)/D(1)."""
    # D satisfies: D' + f*D = 0 in terms of x=ln(a)
    # So D(x) = D(0) * exp(-∫_0^x f(x') dx')
    # D(1)/D(1) = 1, so D(a)/D(1) = exp(-∫_0^{ln(a)} f(x') dx') = exp(∫_{ln(a)}^0 f(x') dx')
    
    a = np.asarray(a, dtype=float)
    f_ref = _ode_reference_f(a, Om0, w0, wa, npts)
    
    # We need to integrate f from ln(a) to 0
    D = np.ones_like(a)
    a_min = max(np.min(a), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    
    # Get f on the grid
    a_grid = np.exp(x_grid)
    f_grid = _ode_reference_f(a_grid, Om0, w0, wa, npts)
    
    # Cumulative integral from 0 to x
    from scipy.integrate import cumulative_trapezoid
    # We want ∫_{x}^{0} f(x') dx' = -∫_0^x f(x') dx'
    # cumulative_trapezoid gives ∫_{x_min}^{x} 
    integral_from_min = cumulative_trapezoid(f_grid, x_grid, initial=0.0)
    # ∫_{x}^{0} f dx' = integral_from_min[-1] - integral_from_min[idx]
    total = integral_from_min[-1]
    
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                int_val = total - integral_from_min[0]
            elif idx >= len(x_grid):
                int_val = 0.0
            else:
                # Interpolate integral
                x0, x1 = x_grid[idx-1], x_grid[idx]
                i0, i1 = integral_from_min[idx-1], integral_from_min[idx]
                t = (x_val - x0) / (x1 - x0)
                int_at_x = i0 + t * (i1 - i0)
                int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

def _compute_error(coeffs, Om0, w0, wa, a_vals):
    """Compute max relative error for f and D."""
    f_approx = _f_of_a(a_vals, coeffs, Om0, w0, wa)
    f_ref = _ode_reference_f(a_vals, Om0, w0, wa)
    
    D_approx = _D_normalized(a_vals, coeffs, Om0, w0, wa)
    D_ref = _ode_reference_D(a_vals, Om0, w0, wa)
    
    # Relative errors, avoid division by zero
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
    deadline = t0 + 80.0  # Leave 10s margin
    
    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try Nelder-Mead
    from scipy.optimize import minimize
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    if time.time() < deadline:
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    
    # Try L-BFGS-B with bounds
    bounds = [(-10, 10)] * 6
    result2 = minimize(_objective, best_x, method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
    if time.time() < deadline:
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    
    # Local refinement
    for _ in range(3):
        if time.time() >= deadline - 5:
            break
        result3 = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'initial_simplex': 0.01 * np.ones(6)})
        if result3.fun < best_val:
            best_val = result3.fun
            best_x = result3.x
        else:
            break
    
    _BEST_COEFFS = list(best_x)
    return _BEST_COEFFS

def construct():
    # Return optimized coefficients
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    else:
        # Fallback: run optimization
        return _optimize()

# Run optimization at import time if possible
try:
    _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
