"""Compact dark-energy growth approximation: fit 6 coefficients to ODE reference."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

# Cache for reference data (generated once)
_REF_CACHE = {}
_DEADLINE = 75.0  # safety margin before 90s kill
_START_TIME = None

def _Om(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    a3 = a**3
    return Om0 * a3 / (Om0 * a3 + (1 - Om0) * np.exp(3 * (1 + w0 + wa) * (1 - a)))

def _f_ref(a, Om0, w0, wa):
    """Reference growth rate from ODE solution (precomputed)."""
    key = (Om0, w0, wa)
    if key not in _REF_CACHE:
        # Solve the growth ODE: d2D/dx2 + [3/(2a) + H'/H] dD/dx - (3/2)Om(a)/a dD/dx = 0
        # Using x = ln(a), D = e^x * dD/dx + ...
        # Standard form: f' + f^2 + (3/2 + dlnH/dx)f - (3/2)Om(a) = 0
        # Integrate backward from a=1 (f(1) = Om(1)^0.55 approx)
        def ode(x, y):
            a = np.exp(x)
            Om_a = _Om(a, Om0, w0, wa)
            # H'/H in terms of x: dlnH/dx = (3/2) * (1 + Om_a + (1-Om_a)*(w0 + wa*(1-a)))
            # Actually: dlnH/dx = (3/2)(1 + Om_a + (1-Om_a)*w(a))
            w_a = w0 + wa * (1 - a)
            dlnH_dx = 1.5 * (1 + Om_a + (1 - Om_a) * w_a)
            f = y[0]
            dfdx = -f*f - (1.5 + dlnH_dx)*f + 1.5*Om_a
            return [dfdx]
        
        # Integrate from x=0 (a=1) backward to x=-5 (a~0.0067)
        x_grid = np.linspace(-5, 0, 2000)
        f_init = _Om(1.0, Om0, w0, wa)**0.55
        sol = solve_ivp(ode, [0, -5], [f_init], t_eval=x_grid, rtol=1e-10, atol=1e-12, method='DOP853')
        
        # Also compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # We need f on a fine grid for integration
        D_norm = np.zeros_like(x_grid)
        for i, xi in enumerate(x_grid):
            # Integrate f from xi to 0
            if xi < 0:
                # Use trapezoidal rule on the solution
                mask = x_grid <= xi
                if np.sum(mask) > 1:
                    x_int = x_grid[mask]
                    f_int = sol.y[0, mask]
                    # Integrate from xi to 0: reverse the segment
                    x_int = x_int[::-1]
                    f_int = f_int[::-1]
                    integral = np.trapz(f_int, x_int)
                    D_norm[i] = np.exp(-integral)
                else:
                    D_norm[i] = 1.0
            else:
                D_norm[i] = 1.0
        
        _REF_CACHE[key] = (x_grid, sol.y[0], D_norm)
    
    return _REF_CACHE[key]

def _evaluate_params(params, Om0, w0, wa, a_grid=None):
    """Evaluate the approximation quality for given parameters."""
    c0, c1, c2, c3, c4, c5 = params
    
    if a_grid is None:
        a_grid = np.linspace(0.05, 1.0, 100)
    
    a = a_grid
    Om_a = _Om(a, Om0, w0, wa)
    q = 1 - Om_a
    u = 1 + w0 + wa * (1 - a)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    
    # Check gamma constraints
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e10
    
    f_approx = Om_a**gamma
    
    # Get reference
    x_grid, f_ref_arr, D_ref_arr = _f_ref(Om0, w0, wa)
    
    # Interpolate reference to our a_grid
    x_eval = np.log(a)
    f_ref_interp = np.interp(x_eval, x_grid, f_ref_arr)
    D_ref_interp = np.interp(x_eval, x_grid, D_ref_arr)
    
    # Compute D_approx
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # We need to integrate f_approx over x
    D_approx = np.zeros_like(a)
    for i, xi in enumerate(x_eval):
        if xi < 0:
            # Integrate f_approx from xi to 0
            x_int = np.linspace(xi, 0, 50)
            a_int = np.exp(x_int)
            Om_int = _Om(a_int, Om0, w0, wa)
            q_int = 1 - Om_int
            u_int = 1 + w0 + wa * (1 - a_int)
            gamma_int = c0 + c1*q_int + c2*q_int**2 + c3*u_int + c4*wa*a_int + c5*u_int*q_int
            f_int = Om_int**gamma_int
            integral = np.trapz(f_int, x_int)
            D_approx[i] = np.exp(-integral)
        else:
            D_approx[i] = 1.0
    
    # Relative errors
    rel_err_f = np.abs(f_approx - f_ref_interp) / np.maximum(np.abs(f_ref_interp), 1e-10)
    rel_err_D = np.abs(D_approx - D_ref_interp) / np.maximum(np.abs(D_ref_interp), 1e-10)
    
    # Worst-case relative error (max over both observables)
    worst_err = np.max(np.concatenate([rel_err_f, rel_err_D]))
    
    return worst_err

def _objective(params, validation_cosmologies):
    """Max relative error across all validation cosmologies."""
    max_err = 0.0
    for Om0, w0, wa in validation_cosmologies:
        err = _evaluate_params(params, Om0, w0, wa)
        if err > max_err:
            max_err = err
    return max_err

# Validation cosmologies (typical CPL models)
_VALIDATION = [
    (0.3, -1.0, 0.0),    # LCDM-like
    (0.3, -1.0, 0.3),    # CPL with wa=0.3
    (0.3, -1.0, -0.5),   # CPL with wa=-0.5
    (0.25, -1.0, 0.0),   # Lower matter density
    (0.35, -1.0, 0.0),   # Higher matter density
    (0.3, -0.9, 0.2),    # Different w0
    (0.3, -1.1, 0.1),    # Different w0
]

def _fit_coefficients():
    """Fit the six coefficients using differential evolution."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Initial bounds
    bounds = [(-10, 10)] * 6
    
    # Start near the seed
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Use differential evolution with a time-limited approach
    def timed_objective(params):
        if time.time() - _START_TIME > _DEADLINE:
            return 1e10
        return _objective(params, _VALIDATION)
    
    try:
        result = differential_evolution(
            timed_objective,
            bounds,
            x0=x0,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            workers=1,
        )
        
        if result.success or result.fun < 1e9:
            # Polish with local optimization
            from scipy.optimize import minimize
            res_local = minimize(
                timed_objective,
                result.x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_local.fun < result.fun:
                result.x = res_local.x
        
        # Clamp to [-10, 10]
        result.x = np.clip(result.x, -10, 10)
        
        return list(result.x)
    
    except Exception:
        return x0

def construct():
    """Return the six fitted coefficients."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check if we have cached results
    if 'fitted' in _REF_CACHE:
        return _REF_CACHE['fitted']
    
    # Time-limited fit
    params = _fit_coefficients()
    
    # Ensure all values are finite and in range
    params = [max(-10, min(10, p)) if np.isfinite(p) else 0.55 for p in params]
    
    _REF_CACHE['fitted'] = params
    return params
