"""Cosmology: compact dark-energy growth approximation — optimized via internal GR reference fit."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 80.0  # Safety margin: 10s before 90s limit

def _om_a(a, om0, wa):
    """Calculate matter density parameter at scale factor a for flat CPL universe."""
    # w(a) = w0 + wa*(1-a), with w0 = -1
    # rho_de ~ a^{-3(1+w0+wa(1-a))}
    # For flat universe: Om_k = 0
    # Om(a) = Om0 * a^{-3} / (Om0 * a^{-3} + (1-Om0) * a^{-3(1+w0+wa(1-a))})
    # w0 = -1 => 1+w0 = 0
    # exponent = -3 * (wa * (1-a))
    # Note: This exact power law is an approximation for CPL if wa is small, 
    # but the task specifies "smooth CPL". The standard CPL density evolution is:
    # rho_de(a) = rho_de0 * a^{-3(1+w0+wa)} * (1 + wa(1-a))^{-1} ... this is complex.
    # However, the standard definition of CPL in many codes uses:
    # w(a) = w0 + wa*(1-a)
    # rho_de(a) = rho_de0 * exp(-3 * integral_1^a (1+w(a'))/a' da')
    # For w0=-1, w(a) = -1 + wa(1-a).
    # 1+w(a) = wa(1-a).
    # integral_1^a wa(1-a')/a' da' = wa * integral_1^a (1/a' - 1) da' = wa * [ln a' - a']_1^a = wa * (ln a - a - (0 - 1)) = wa * (ln a - a + 1).
    # So rho_de(a) = rho_de0 * exp(-3 * wa * (ln a - a + 1)).
    
    de_exp = -3.0 * wa * (np.log(a) - a + 1.0)
    rho_de = np.exp(de_exp)
    rho_m = om0 * a**(-3.0)
    return rho_m / (rho_m + (1.0 - om0) * rho_de)

def _solve_gr_ode(om0, wa, a_grid):
    """Solve the GR ODE for f(a) and D(a)."""
    # We solve for f(x) where x = ln(a)
    # df/dx + f(f+1) + 1.5 * Om(x) = 0
    # D(x) = D(1) * exp(integral f dx)
    
    x_grid = np.log(a_grid)
    x_start = 0.0 # a=1
    
    def ode_func(x, y):
        a = np.exp(x)
        om = _om_a(a, om0, wa)
        f = y[0]
        df_dx = -f * (f + 1.0) - 1.5 * om
        dD_dx = f * y[1]
        return [df_dx, dD_dx]
    
    # Integrate backwards from a=1 to a_min
    # Initial condition at a=1: f(1) ~ 0.55 * Om0^0.55, D(1)=1
    f_init = 0.55 * (om0 ** 0.55)
    y_init = [f_init, 1.0]
    
    # We need to integrate from x=0 down to x_min
    # solve_ivp integrates forward, so we reverse the grid or use t_span (x_min, 0) and reverse results
    x_min = x_grid[0]
    x_max = 0.0
    
    # Define a grid for integration
    t_eval = np.linspace(x_min, x_max, len(a_grid))
    
    try:
        sol = solve_ivp(ode_func, (x_min, x_max), y_init, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        if not sol.success:
            # Fallback to simpler integration if fails
            return None, None
        f_gr = sol.y[0][::-1] # Reverse to match a_grid order (a_min to 1)
        D_gr = sol.y[1][::-1]
        return f_gr, D_gr
    except Exception:
        return None, None

def _compute_errors(coeffs, om0, wa, a_grid, f_gr, D_gr):
    """Compute relative errors for f and D."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Calculate q, u, gamma
    om_a = _om_a(a_grid, om0, wa)
    q = 1.0 - om_a
    u = 1.0 + wa * (1.0 - a_grid)
    
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_grid + c5 * u * q
    
    # Check gamma bounds
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e6 # Penalize out of bounds
    
    f_app = om_a ** gamma
    
    # Calculate D_app
    # D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    # Let x = ln(a). Integral from ln(a) to 0 of f(e^x) dx.
    # We can approximate this integral numerically using the grid.
    x = np.log(a_grid)
    # f_app is defined on a_grid. We need f(e^x) for x in [ln(a), 0].
    # Since we have the grid, we can use cumulative trapezoidal integration.
    # Note: x is decreasing from 0 to ln(a_min).
    # We want integral from x to 0.
    # Let's compute integral from 0 to x_i, then negate.
    # Integral_0^{x_i} f(e^t) dt.
    # Using trapezoid rule on the grid points.
    
    # Sort x to be increasing for integration? 
    # a_grid is typically increasing (a_min ... 1). x is increasing (ln(a_min) ... 0).
    # We want I(x_i) = integral_{x_i}^0 f(t) dt = - integral_0^{x_i} f(t) dt.
    # Let's compute cumulative integral from start (x_min) to end (0).
    # C_j = integral_{x_0}^{x_j} f(t) dt.
    # Then integral_{x_i}^0 = C_{end} - C_i.
    
    # Ensure x is sorted
    # a_grid is usually sorted.
    x_sorted = x
    f_sorted = f_app
    
    # Trapezoidal integration
    dx = np.diff(x_sorted)
    avg_f = 0.5 * (f_sorted[:-1] + f_sorted[1:])
    integrals = np.cumsum(dx * avg_f)
    integrals = np.insert(integrals, 0, 0.0)
    
    # Total integral from x_min to 0
    total_int = integrals[-1]
    
    # Integral from x_i to 0 is total_int - integrals[i]
    int_x_to_0 = total_int - integrals
    
    D_app = np.exp(-int_x_to_0)
    
    # Relative errors
    # Avoid division by zero
    mask = (np.abs(f_gr) > 1e-6) & (np.abs(D_gr) > 1e-6)
    
    if not np.any(mask):
        return 1e6
        
    err_f = np.abs((f_app[mask] - f_gr[mask]) / f_gr[mask])
    err_D = np.abs((D_app[mask] - D_gr[mask]) / D_gr[mask])
    
    max_err = np.max(np.concatenate([err_f, err_D]))
    return max_err

def construct():
    """Return six coefficients for the growth-index approximation."""
    global _DEADLINE
    
    # Validation cosmologies: representative range
    # Om0 in [0.2, 0.4], wa in [-0.5, 0.5]
    om0s = [0.25, 0.30, 0.35]
    was = [-0.3, -0.1, 0.0, 0.1, 0.3]
    
    # Scale factor grid
    a_min = 0.05
    a_max = 1.0
    n_points = 50
    a_grid = np.linspace(a_min, a_max, n_points)
    
    # Pre-compute GR references
    # This might take some time, so we do it once.
    refs = {}
    for om0 in om0s:
        for wa in was:
            f_gr, D_gr = _solve_gr_ode(om0, wa, a_grid)
            if f_gr is not None:
                refs[(om0, wa)] = (f_gr, D_gr)
    
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    def objective(coeffs):
        if time.time() > _DEADLINE:
            return 1e9
        max_err = 0.0
        for key, (f_gr, D_gr) in refs.items():
            err = _compute_errors(coeffs, key[0], key[1], a_grid, f_gr, D_gr)
            if err > max_err:
                max_err = err
            if err > 1e6: # Early exit if invalid
                return 1e6
        return max_err
    
    # Initial guess
    x0 = [0.55, 0.05, -0.02, 0.03, 0.0, -0.01]
    
    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for small dimensions
    best_coeffs = x0
    best_score = objective(x0)
    
    # Simple iterative optimization with time check
    # We can run a few steps of minimize
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if result.success and result.fun < best_score:
            best_coeffs = result.x
            best_score = result.fun
    except Exception:
        pass
        
    # Check if we have time for a second refinement with tighter bounds or different method
    if time.time() < _DEADLINE - 5.0:
        try:
            result2 = minimize(objective, best_coeffs, method='BFGS', 
                               options={'maxiter': 50, 'gtol': 1e-5})
            if result2.fun < best_score:
                best_coeffs = result2.x
                best_score = result2.fun
        except Exception:
            pass

    # Ensure coefficients are finite and within [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    if not np.all(np.isfinite(best_coeffs)):
        return [0.55, 0.05, -0.02, 0.03, 0.0, -0.01]
        
    return list(best_coeffs)
# EVOLVE-BLOCK-END
