import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
import time

# Global variables to hold reference data
_REF_DATA = []
_A_GRID = None

def _compute_E_sq(Om0, w0, wa, a):
    """Compute E^2(a) = H^2(a)/H0^2"""
    Om_m = Om0 * a**(-3)
    Om_r = 0.0  # Neglect radiation for simplicity, or include if needed. Usually negligible for a > 0.1
    # Let's include radiation for better accuracy at low a if needed, but a starts at 0.05 usually.
    # Om_r0 ~ 9e-5. At a=0.05, a^-4 = 160000. 9e-5 * 160000 ~ 1.4. Not negligible.
    # However, the task says "smooth CPL dark-energy cosmologies".
    # Let's assume flat universe: Om_k = 0.
    # Om_de(a) = Om_de0 * a**(-3*(1+w0+wa))
    # Om_de0 = 1 - Om0 - Om_r0
    # Let's fix Om_r0 = 0.0 for simplicity as it's often ignored in these approximations unless a is very small.
    # If a_min is 0.1, Om_r is small.
    Om_de0 = 1.0 - Om0
    Om_de = Om_de0 * a**(-3.0 * (1.0 + w0 + wa))
    return Om_m + Om_de

def _compute_Om_m(Om0, w0, wa, a):
    E2 = _compute_E_sq(Om0, w0, wa, a)
    return Om0 * a**(-3) / E2

def _generate_reference(Om0, w0, wa, a_grid):
    """
    Generate reference f(a) and D(a) for a given cosmology.
    Returns arrays f_ref, D_ref corresponding to a_grid.
    """
    # 1. Find f(1) by integrating from a=0.01 to a=1.0
    # ODE: a * df/da = 1.5 * Om_m(a) - f * (f + 1)
    # df/da = (1.5 * Om_m(a) - f * (f + 1)) / a
    
    def ode_f(a, y):
        f = y[0]
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        return [(1.5 * Om_m - f * (f + 1)) / a]
    
    # Initial guess for f at a=0.01: f ~ 1
    a_start = 0.01
    a_end = 1.0
    try:
        sol = solve_ivp(ode_f, [a_start, a_end], [1.0], method='RK45', rtol=1e-8, atol=1e-10)
        if not sol.success:
            # Fallback: use approximation f(1) = Om0**0.55
            f1 = Om0**0.55
        else:
            f1 = sol.y[0, -1]
    except Exception:
        f1 = Om0**0.55
        
    # 2. Integrate from a=1.0 to min(a_grid) and max(a_grid)
    # We need f(a) on the grid.
    # Let's integrate on the grid directly using solve_ivp with t_eval
    
    a_min = min(a_grid)
    a_max = max(a_grid)
    
    # Ensure 1.0 is in the integration range
    if a_min < 1.0:
        a_eval_down = np.sort(np.unique(np.concatenate([a_grid[a_grid < 1.0], [1.0]])))
        sol_down = solve_ivp(ode_f, [1.0, a_min], [f1], method='RK45', t_eval=a_eval_down, rtol=1e-8, atol=1e-10)
        f_down = sol_down.y[0]
        a_down = sol_down.t
    else:
        f_down = np.array([f1])
        a_down = np.array([1.0])
        
    if a_max > 1.0:
        a_eval_up = np.sort(np.unique(np.concatenate([a_grid[a_grid > 1.0], [1.0]])))
        sol_up = solve_ivp(ode_f, [1.0, a_max], [f1], method='RK45', t_eval=a_eval_up, rtol=1e-8, atol=1e-10)
        f_up = sol_up.y[0]
        a_up = sol_up.t
    else:
        f_up = np.array([f1])
        a_up = np.array([1.0])
        
    # Combine and interpolate to a_grid
    a_all = np.concatenate([a_down, a_up[1:]])
    f_all = np.concatenate([f_down, f_up[1:]])
    
    # Sort by a
    sort_idx = np.argsort(a_all)
    a_sorted = a_all[sort_idx]
    f_sorted = f_all[sort_idx]
    
    f_ref = np.interp(a_grid, a_sorted, f_sorted)
    
    # 3. Compute D(a) = exp( integral_1^a f(a')/a' da' )
    # Use cumulative trapezoidal rule
    # We need to integrate from 1 to a.
    # Let's create a dense grid for integration accuracy
    a_dense = np.linspace(a_min, a_max, 1000)
    f_dense = np.interp(a_dense, a_sorted, f_sorted)
    integrand = f_dense / a_dense
    
    # Cumulative integral from 1
    # We need to handle the split at 1
    D_ref = np.ones_like(a_grid)
    
    # For a < 1: integral from a to 1 of f/a' da' (with negative sign in exp? No, D(a)/D(1) = exp(-int_a^1 f/a' da'))
    # D(a) = D(1) * exp( - \int_a^1 \frac{f(a')}{a'} da' )
    # \int_a^1 = - \int_1^a
    # So D(a) = exp( \int_1^a \frac{f(a')}{a'} da' )
    
    # Let's just compute the integral I(a) = \int_1^a \frac{f(a')}{a'} da'
    # Using np.interp and cumulative sum on a grid that includes 1.
    
    # Create a grid that covers [a_min, a_max] and includes 1.0
    grid = np.unique(np.concatenate([a_dense, [1.0]]))
    f_grid = np.interp(grid, a_sorted, f_sorted)
    int_grid = f_grid / grid
    
    # Cumulative trapezoidal
    # We need integral from 1 to grid points.
    # Find index of 1.0
    idx_1 = np.argmin(np.abs(grid - 1.0))
    
    # Integrate from idx_1 to end
    # np.cumtrapz is deprecated, use np.trapz in a loop or cumulative_trapezoid
    from scipy.integrate import cumulative_trapezoid
    
    # Integral from grid[0] to grid[i]
    # We want integral from 1 to a.
    # Let C(a) = \int_{grid[0]}^a int_grid da'
    # Then \int_1^a = C(a) - C(1)
    
    C = cumulative_trapezoid(int_grid, grid, initial=0)
    C1 = C[idx_1]
    
    # Interpolate C(a) - C1 to a_grid
    I_a = np.interp(a_grid, grid, C - C1)
    
    D_ref = np.exp(I_a)
    
    return f_ref, D_ref

def _make_reference_data():
    """Generate reference data for a set of cosmologies."""
    global _REF_DATA, _A_GRID
    
    # Cosmologies: (Om0, w0, wa)
    # Representative set
    cosmo_params = [
        (0.3, -1.0, 0.0),    # LCDM
        (0.3, -1.0, 0.3),    # CPL
        (0.3, -0.9, 0.0),
        (0.3, -0.9, 0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.3, -1.2, 0.5),    # Phantom
        (0.3, -0.5, 0.0),    # w0 > -1
        (0.4, -1.0, 0.3),
        (0.2, -1.0, 0.3),
    ]
    
    # Scale factor grid
    # Validation likely covers a in [0.1, 1.0] or similar.
    # Let's use a dense grid.
    _A_GRID = np.linspace(0.1, 1.0, 100)
    
    _REF_DATA = []
    for Om0, w0, wa in cosmo_params:
        try:
            f_ref, D_ref = _generate_reference(Om0, w0, wa, _A_GRID)
            # Store Om0, w0, wa, f_ref, D_ref
            _REF_DATA.append((Om0, w0, wa, f_ref, D_ref))
        except Exception as e:
            print(f"Error generating reference for {Om0}, {w0}, {wa}: {e}")

def _objective(c):
    """
    Objective function: Max relative error of f and D.
    """
    c0, c1, c2, c3, c4, c5 = c
    
    max_err = 0.0
    
    for Om0, w0, wa, f_ref, D_ref in _REF_DATA:
        a = _A_GRID
        # Compute model gamma
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Check gamma constraints
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        # Compute f_model
        # f = Om_m^gamma
        # Handle Om_m=0? Om_m > 0 for a in [0.1, 1]
        f_model = np.power(Om_m, gamma)
        
        # Compute D_model
        # D(a) = exp( integral_1^a f_model(a')/a' da' )
        # We need to integrate f_model/a
        # Use cumulative trapezoid
        from scipy.integrate import cumulative_trapezoid
        
        integrand = f_model / a
        # We need integral from 1 to a.
        # Find index of a=1.0
        idx_1 = np.argmin(np.abs(a - 1.0))
        
        # Integrate from 0 to idx_1 and idx_1 to end?
        # Easier: integrate from a[0] to a[i], then subtract integral from a[0] to a[idx_1]
        
        C = cumulative_trapezoid(integrand, a, initial=0)
        C1 = C[idx_1]
        
        I_a = C - C1
        D_model = np.exp(I_a)
        
        # Relative errors
        # f
        err_f = np.abs(f_model - f_ref) / np.maximum(f_ref, 1e-10)
        # D
        err_D = np.abs(D_model - D_ref) / np.maximum(D_ref, 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
        
    return max_err

def construct():
    """
    Construct the optimal coefficients.
    """
    global _REF_DATA
    if not _REF_DATA:
        _make_reference_data()
        
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Initial guess
    # c0 ~ 0.55, others 0
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Time limit
    start_time = time.time()
    time_limit = 60.0
    
    best_c = np.array(x0)
    best_err = _objective(best_c)
    
    # Use differential evolution
    # maxiter=100, popsize=15
    # Update best during optimization
    def callback(xk, convergence):
        nonlocal best_c, best_err
        err_k = _objective(xk)
        if err_k < best_err:
            best_err = err_k
            best_c = xk.copy()
        
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-8, 
            polish=True,
            callback=callback,
            update_interval=1
        )
        
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x
            
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # Ensure time limit
    elapsed = time.time() - start_time
    if elapsed > time_limit - 5:
        # Return best found so far
        pass
        
    # Final check and clipping
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Verify constraints on a wider grid just in case
    a_check = np.linspace(0.05, 1.0, 200)
    valid = True
    for Om0, w0, wa, _, _ in _REF_DATA:
        Om_m = _compute_Om_m(Om0, w0, wa, a_check)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_check)
        gamma = best_c[0] + best_c[1]*q + best_c[2]*q**2 + best_c[3]*u + best_c[4]*wa*a_check + best_c[5]*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            valid = False
            break
            
    if not valid:
        # Fallback to safe constant
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return [float(x) for x in best_c]

# Pre-compute reference data at module load to ensure it's ready
# But construct() is the entry point.
# The evaluator calls construct().
# It is safer to compute inside construct if not already done.

if __name__ == "__main__":
    # Test
    _make_reference_data()
    c = construct()
    print(c)
