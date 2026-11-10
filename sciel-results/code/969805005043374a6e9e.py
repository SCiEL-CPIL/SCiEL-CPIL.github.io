"""
Improved growth-index approximation solver.
Uses numerical ODE solution as reference and robust optimization.
"""
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def _compute_om_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    # E(a)^2 = Om0/a^3 + (1-Om0)*(1-a)^(-3*(1+w0+wa))
    # Note: for a close to 1, (1-a) can be 0, handle carefully
    de_exponent = -3.0 * (1.0 + w0 + wa)
    if a >= 1.0:
        de_factor = 1.0
    else:
        one_minus_a = max(1.0 - a, 1e-15)
        de_factor = one_minus_a ** de_exponent
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    Om_a = (Om0 / (a ** 3)) / E2
    return Om_a

def _solve_growth_reference(Om0, w0, wa, a_grid):
    """
    Solve the growth ODE numerically to get f_ref and D_ref at given a_grid.
    ODE: df/dlna = -f^2 - (1.5*Om(a) + 0.5)f + 1.5*Om(a)
    Initial condition: f(1) = Om(1)^0.55 ~ 0.55 (standard approximation)
    Actually, f(1) is determined by the model. For LambdaCDM, f(1) ~ 0.54.
    We can integrate backwards from a=1.
    """
    # We need f at a=1. A good initial guess is Om0^0.55.
    f_init = Om0 ** 0.55
    
    # Define the ODE for f as a function of x = ln(a)
    # a = exp(x)
    def ode_rhs(x, y):
        f = y[0]
        a = np.exp(x)
        Om_a = _compute_om_a(a, Om0, w0, wa)
        # df/dx = -f^2 - (1.5*Om_a + 0.5)*f + 1.5*Om_a
        return [-f**2 - (1.5 * Om_a + 0.5) * f + 1.5 * Om_a]

    # Integrate from x=0 (a=1) to x_min = ln(a_min)
    # a_grid is sorted ascending, so x_grid is sorted ascending.
    # We want to solve for x in [ln(a_min), 0].
    # solve_ivp integrates from t0 to tf.
    # So we integrate from 0 to ln(a_min).
    
    x_min = np.log(a_grid[0])
    
    # We need to evaluate at specific points a_grid.
    # t_eval should be in the interval [0, x_min] if x_min < 0.
    # Since a < 1, x_min < 0.
    # So we integrate from 0 to x_min.
    
    sol = solve_ivp(ode_rhs, [0.0, x_min], [f_init], t_eval=np.log(a_grid), 
                    method='RK45', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        # Fallback to simple approximation if solver fails
        f_ref = np.array([_compute_om_a(a, Om0, w0, wa) ** 0.55 for a in a_grid])
        # Approximate D
        D_ref = np.ones_like(f_ref)
        return f_ref, D_ref

    f_ref = sol.y[0]
    
    # Compute D(a)/D(1) = exp(-int_{ln a}^0 f(e^x) dx)
    # D(a) = exp(- \int_{x}^{0} f(x') dx')
    # Let I(x) = \int_{x}^{0} f(x') dx'
    # We can compute the cumulative integral from x_min to 0.
    # np.cumsum integrates from start to end.
    # We have x_grid = log(a_grid) which is [x_min, ..., 0].
    # We want \int_{x}^{0} f(x') dx'.
    # Let F(x) = \int_{x}^{0} f(x') dx'.
    # F(0) = 0.
    # F(x) = - \int_{0}^{x} f(x') dx'.
    
    # Use cumulative trapezoidal rule
    x_vals = np.log(a_grid)
    # x_vals is sorted ascending: [x_min, ..., 0]
    # We want integral from x to 0.
    # Let's compute integral from 0 to x, then negate.
    # scipy.integrate.cumulative_trapezoid(y, x, initial=0) gives \int_{x[0]}^{x[i]} y dx.
    # This is \int_{x_min}^{x[i]} f dx.
    # We want \int_{x[i]}^{0} f dx = \int_{x_min}^{0} f dx - \int_{x_min}^{x[i]} f dx.
    
    from scipy.integrate import cumulative_trapezoid
    # f_ref corresponds to x_vals
    # Integrate f_ref with respect to x_vals
    int_f = cumulative_trapezoid(f_ref, x_vals, initial=0)
    # int_f[i] = \int_{x_min}^{x_vals[i]} f(x') dx'
    # Total integral from x_min to 0 is int_f[-1]
    total_int = int_f[-1]
    # We want \int_{x_vals[i]}^{0} f(x') dx' = total_int - int_f[i]
    # Note: x_vals[i] >= x_min.
    # Since f > 0 and x increases, int_f increases.
    # \int_{x}^{0} f dx = \int_{x}^{x_{max}} f dx.
    # My int_f is \int_{x_{min}}^{x} f dx.
    # So \int_{x}^{0} f dx = \int_{x_{min}}^{0} f dx - \int_{x_{min}}^{x} f dx = total_int - int_f.
    
    exponents = -(total_int - int_f)
    D_ref = np.exp(exponents)
    
    return f_ref, D_ref

def _growth_index_error(coeffs):
    """Compute the worst relative error for a given set of 6 coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Test cosmologies: (Om0, w0, wa)
    # Include a variety to ensure robustness
    test_models = [
        (0.3, -1.0, 0.0),    # LCDM
        (0.3, -0.9, -0.3),   # CPL
        (0.3, -1.2, 0.5),    # CPL
        (0.3, -0.8, 0.0),    # Quintessence-like
        (0.2, -1.0, 0.0),    # Low Om
        (0.4, -1.0, 0.0),    # High Om
        (0.3, -1.0, -1.0),   # Strong wa
        (0.3, -1.0, 1.0),    # Strong wa
        (0.3, -0.5, 0.0),    # Mild DE
        (0.3, -1.5, 0.5),    # Phantom-like
    ]
    
    # Dense grid for a
    # Avoid a=0 and a=1 exactly for numerical stability if needed, but a=1 is fine.
    a_values = np.linspace(0.05, 1.0, 50)
    
    max_rel_err = 0.0
    
    for Om0, w0, wa in test_models:
        # Get reference solutions
        f_ref, D_ref = _solve_growth_reference(Om0, w0, wa, a_values)
        
        for i, a in enumerate(a_values):
            if a <= 0:
                continue
                
            Om_a = _compute_om_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                # Penalize out-of-bounds gamma
                return 1000.0
            
            f_approx = Om_a ** gamma
            f_ref_val = f_ref[i]
            
            rel_err_f = abs(f_approx - f_ref_val) / max(abs(f_ref_val), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
            
            # Compute D_approx
            # D(a)/D(1) = exp(-\int_{\ln a}^0 f_approx(e^x) dx)
            # f_approx(e^x) = Om_a(e^x)^gamma(e^x)
            # We need to integrate this numerically.
            # To save time, we can use a simple trapezoidal rule on the grid points.
            # However, gamma varies with a.
            # Let's precompute the integrand for this specific a? No, that's O(N^2).
            # We can approximate the integral using the values on the grid.
            
            # Let's compute the integral for each a using the grid points between a and 1.
            # This is expensive if done naively for every a.
            # Alternative: Compute D_approx for all a in the grid at once?
            # The integral is from ln(a) to 0.
            # We can compute the cumulative integral of the approximation f_approx_grid.
            
            # Let's restructure: compute f_approx for all a_values first.
            # Then compute D_approx for all a_values.
            
            # Break out of inner loop to do this vectorized?
            # No, let's just compute it here for simplicity, but it might be slow.
            # 50 points * 10 models * 60s limit.
            # 500 evaluations per error function call.
            # Each evaluation involves an integral of ~50 points.
            # 500 * 50 = 25,000 operations. Very fast.
            
            # We need the integral of f_approx from ln(a) to 0.
            # We have a_values sorted ascending.
            # i corresponds to a_values[i].
            # We need to integrate f_approx from a_values[i] to a_values[-1] (which is 1.0).
            # f_approx_j = Om(a_j)^gamma_j.
            
            # Let's precompute f_approx for the whole grid for this model to avoid recomputing Om_a.
            # But we are already in the loop.
            
            # Let's just do a simple trapezoidal integration from a to 1.
            # Indices i to end.
            a_sub = a_values[i:]
            f_sub = np.array([_compute_om_a(aa, Om0, w0, wa) ** (_compute_om_a(aa, Om0, w0, wa) and 0) for aa in a_sub]) # Placeholder
            # This is getting messy.
            
            # Let's optimize the D calculation.
            # Compute f_approx for all a_values in the grid for this model.
            # Do this outside the 'for a in a_values' loop.
            
            # Refactor:
            # Move the f_approx calculation out.
            
            pass # This block is replaced by the refactored logic below

    # Refactored Loop for Efficiency
    max_rel_err = 0.0
    for Om0, w0, wa in test_models:
        f_ref, D_ref = _solve_growth_reference(Om0, w0, wa, a_values)
        
        # Compute f_approx for all a in a_values
        Om_a_grid = np.array([_compute_om_a(a, Om0, w0, wa) for a in a_values])
        q_grid = 1.0 - Om_a_grid
        u_grid = 1.0 + w0 + wa * (1.0 - a_values)
        
        gamma_grid = c0 + c1 * q_grid + c2 * q_grid**2 + c3 * u_grid + c4 * wa * a_values + c5 * u_grid * q_grid
        
        # Check bounds
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            return 1000.0
            
        f_approx_grid = Om_a_grid ** gamma_grid
        
        # Compute D_approx_grid
        # D(a)/D(1) = exp(-\int_{\ln a}^0 f_approx(e^x) dx)
        # Let x = ln(a).
        # We need \int_{x}^{0} f_approx(x') dx'.
        # Use cumulative trapezoid on x_grid = ln(a_values).
        
        x_grid = np.log(a_values)
        # Integrate f_approx_grid with respect to x_grid
        # cumulative_trapezoid(y, x, initial=0) -> \int_{x[0]}^{x[i]} y dx
        from scipy.integrate import cumulative_trapezoid
        int_f_approx = cumulative_trapezoid(f_approx_grid, x_grid, initial=0)
        total_int_approx = int_f_approx[-1]
        # \int_{x[i]}^{0} f dx = total_int_approx - int_f_approx[i]
        exponents_approx = -(total_int_approx - int_f_approx)
        D_approx_grid = np.exp(exponents_approx)
        
        # Calculate errors
        for i in range(len(a_values)):
            rel_err_f = abs(f_approx_grid[i] - f_ref[i]) / max(abs(f_ref[i]), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
            
            rel_err_D = abs(D_approx_grid[i] - D_ref[i]) / max(abs(D_ref[i]), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_D)
            
    return max_rel_err * 100

def construct():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    best_x = x0
    best_err = _growth_index_error(x0)
    
    # Multi-start optimization
    starts = [
        x0,
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
        
        try:
            # Use L-BFGS-B with bounds
            bounds = [(-10, 10)] * 6
            result = minimize(_growth_index_error, start, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
                
        except Exception:
            pass
            
    # Final polish with Nelder-Mead if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(_growth_index_error, best_x, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except Exception:
            pass
            
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
