"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _Om(a, Om0, w0, wa):
    """Matter density parameter as function of scale factor a."""
    # Om(a) = Om0 * a^-3 / (Om0 * a^-3 + (1-Om0) * exp(3*(1+w0+wa*(1-a)) * (1-a) / (1+w0+wa)))
    # Actually, for CPL: Om_DE(a) = (1-Om0) * a^(-3*(1+w0+wa*(1-a)))
    # Wait, the standard form: Om_DE(a) = (1-Om0) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # Let me use the correct CPL formula:
    # rho_DE(a) = rho_DE(1) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # So Om_DE(a) = (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a)) / (Om0*a^(-3) + (1-Om0)*a^(-3*(1+w0))*exp(-3*wa*(1-a)))
    # But the task says u = 1+w0+wa*(1-a), and the integral uses this.
    # Let me re-read: the task defines u = 1+w0+wa*(1-a)
    # The standard CPL: w(a) = w0 + wa*(1-a)
    # rho_DE(a) = rho_DE(1) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # Om_DE(a) = (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a)) / (Om0*a^(-3) + (1-Om0)*a^(-3*(1+w0))*exp(-3*wa*(1-a)))
    # But for the ODE reference, I need the actual Om(a) from the Friedmann equation.
    
    a = np.asarray(a, dtype=float)
    if np.isscalar(a) or a.ndim == 0:
        a = np.array([a])
    scalar_input = (np.isscalar(a) or a.ndim == 0)
    
    # Use the standard CPL density evolution
    # Om_m(a) = Om0 * a^(-3) / E^2(a)
    # E^2(a) = Om0*a^(-3) + (1-Om0)*a^(-3*(1+w0))*exp(-3*wa*(1-a))
    # But actually, the task's integral formula suggests a simpler relation.
    # Let me use the exact CPL formula.
    
    a_safe = np.maximum(a, 1e-10)
    Om_m = Om0 * a_safe**(-3)
    Om_de = (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(-3*wa*(1-a_safe))
    E2 = Om_m + Om_de
    Om_a = Om_m / E2
    if scalar_input:
        return float(Om_a[0])
    return Om_a

def _gamma_from_coeffs(c, a, Om0, w0, wa):
    """Compute gamma from coefficients and cosmological parameters."""
    Om_a = _Om(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_from_gamma(gamma, Om0, w0, wa, a):
    """Compute f = Om(a)^gamma."""
    Om_a = _Om(a, Om0, w0, wa)
    Om_a = np.maximum(Om_a, 1e-30)
    return Om_a ** gamma

def _numerical_reference(Om0, w0, wa, a_eval, n_steps=10000):
    """
    Compute f and normalized D using numerical GR ODE integration.
    Uses the growth equation: f' + f^2 + (3/2)H/H' f - (1/2)Om_m' / H/H' = 0
    In terms of x = ln(a): df/dx + f^2 + (3/2) dlnH/dx f - (1/2) dln(Om_m)/dx = 0
    dlnH/dx = (3/2)(1 + w_eff(a)) where w_eff = -1 + Om_m/(3*(1+...))
    Actually: dlnE/dx = (3/2)(1 + w(a)) for CPL... no.
    E^2 = Om_m + Om_de
    dlnE^2/dx = 2*E*E' = d(Om_m)/dx + d(Om_de)/dx
    d(Om_m)/dx = -3*Om_m (since Om_m ~ a^-3)
    d(Om_de)/dx = Om_de * (-3*(1+w0) + 3*wa) = Om_de * 3*wa - 3*(1+w0)*Om_de
    Wait: Om_de(a) = (1-Om0)*a^(-3(1+w0))*exp(-3*wa*(1-a))
    d(Om_de)/dx = Om_de * [-3*(1+w0) + 3*wa] = Om_de * 3*(wa - 1 - w0)
    
    So dlnE^2/dx = -3*Om_m + 3*(wa-1-w0)*Om_de
    dlnE/dx = (1/2) * dlnE^2/dx / E^2 = (1/2)*(-3*Om_m + 3*(wa-1-w0)*Om_de)
    = (3/2)*(-Om_m + (wa-1-w0)*Om_de)
    
    dln(Om_m)/dx = -3
    
    Growth ODE in x: df/dx + f^2 + (3/2)*dlnE/dx * f - (1/2)*dln(Om_m)/dx = 0
    df/dx = -f^2 - (3/2)*dlnE/dx * f + (1/2)*3
    = -f^2 - (3/2)*dlnE/dx * f + 3/2
    """
    a_eval = np.asarray(a_eval, dtype=float)
    scalar_input = (a_eval.ndim == 0)
    if scalar_input:
        a_eval = np.array([a_eval])
    
    # We need to integrate from x=0 (a=1) to x=ln(a) for each a
    # For a < 1, ln(a) < 0, so we integrate backwards
    # Let's integrate from x=0 to x_min = min(ln(a_eval))
    
    x_eval = np.log(a_eval)
    x_min = min(0.0, np.min(x_eval))
    x_max = 0.0
    
    # Create a grid from x_min to x_max
    n_steps = 5000
    x_grid = np.linspace(x_min, x_max, n_steps)
    
    # Initial condition at x=0: f(0) ~ Om0^0.55 (approximate), but better to use f(0) = Om0^0.55
    # Actually, for the reference, we should use a good initial condition.
    # f(0) for a flat universe: f(0) = Om0^0.55 is standard approximation
    # But for the GR reference, we can use f(0) = 0.55*Om0^0.55... no, f = Om^gamma
    # At a=1, Om(1)=Om0, so f(1) = Om0^gamma. For the reference, we need the true f(1).
    # In a flat universe, f(1) = Om0^0.55 is a good approximation, but let's use a slightly better one.
    # Actually, the exact f(1) for a flat universe can be computed, but for our purposes,
    # we'll use f(1) = Om0^0.55 as initial condition for the ODE.
    
    f_init = Om0**0.55
    
    # Integrate ODE: df/dx = -f^2 - (3/2)*dlnE/dx * f + 3/2
    # where dlnE/dx = (3/2)*(-Om_m + (wa-1-w0)*Om_de)
    
    def ode_rhs(x, f):
        a = np.exp(x)
        Om_m = Om0 * a**(-3)
        Om_de = (1 - Om0) * a**(-3*(1+w0)) * np.exp(-3*wa*(1-a))
        dlnE_dx = (3/2)*(-Om_m + (wa - 1 - w0)*Om_de)
        return -f**2 - (3/2)*dlnE_dx * f + 3/2
    
    # Use simple RK4 integration
    dx = x_grid[1] - x_grid[0] if n_steps > 1 else 0
    f_vals = np.zeros(n_steps)
    f_vals[-1] = f_init  # at x_max = 0
    
    # Integrate backwards from x=0 to x_min
    for i in range(n_steps - 2, -1, -1):
        x = x_grid[i+1]
        f = f_vals[i+1]
        k1 = ode_rhs(x, f)
        k2 = ode_rhs(x - dx/2, f - dx*k1/2)
        k3 = ode_rhs(x - dx/2, f - dx*k2/2)
        k4 = ode_rhs(x - dx, f - dx*k3)
        f_vals[i] = f - dx*(k1 + 2*k2 + 2*k3 + k4)/6
    
    # Interpolate f at x_eval points
    f_ref = np.interp(x_eval, x_grid, f_vals)
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # = exp(-integral of f over x from ln(a) to 0)
    # For a < 1: ln(a) < 0, integral from ln(a) to 0
    # We have f_vals on x_grid from x_min to x_max=0
    # integral from x_eval[i] to 0 of f dx
    
    # Compute cumulative integral from x_min to each point
    # integral from x_min to x = cumsum of f * dx
    cum_int = np.zeros(n_steps)
    for i in range(1, n_steps):
        cum_int[i] = cum_int[i-1] + (f_vals[i] + f_vals[i-1])/2 * (x_grid[i] - x_grid[i-1])
    
    # integral from x_eval[i] to 0 = cum_int at x=0 - cum_int at x_eval[i]
    # cum_int at x=0 is cum_int[-1]
    int_to_0 = np.interp(x_eval, x_grid, cum_int)
    integral = cum_int[-1] - int_to_0  # integral from x_eval to 0
    
    D_ratio = np.exp(-integral)
    
    if scalar_input:
        return float(f_ref[0]), float(D_ratio[0])
    return f_ref, D_ratio

def _compute_error(c, test_cosmos, a_values):
    """Compute the worst relative error across all test cosmologies and a values."""
    max_err = 0.0
    for (Om0, w0, wa) in test_cosmos:
        f_ref, D_ref = _numerical_reference(Om0, w0, wa, a_values, n_steps=3000)
        
        a_arr = np.asarray(a_values, dtype=float)
        gamma = _gamma_from_coeffs(c, a_arr, Om0, w0, wa)
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6
        
        f_approx = _f_from_gamma(gamma, Om0, w0, wa, a_arr)
        
        # Relative errors
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        err_D = np.abs(np.exp(-np.log(D_ref)) - np.exp(-np.log(D_ref)))  # placeholder
        
        # Actually, I need to compute D_approx properly
        # D_approx(a)/D_approx(1) = exp(-integral from ln(a) to 0 of f_approx(exp(x)) dx)
        # Let me compute this numerically using the same grid
        
        x_eval = np.log(a_arr)
        x_min = min(0.0, np.min(x_eval))
        n_steps = 3000
        x_grid = np.linspace(x_min, 0.0, n_steps)
        
        # Compute f_approx on x_grid
        a_grid = np.exp(x_grid)
        gamma_grid = _gamma_from_coeffs(c, a_grid, Om0, w0, wa)
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            return 1e6
        f_approx_grid = _f_from_gamma(gamma_grid, Om0, w0, wa, a_grid)
        
        # Cumulative integral
        cum_int = np.zeros(n_steps)
        for i in range(1, n_steps):
            cum_int[i] = cum_int[i-1] + (f_approx_grid[i] + f_approx_grid[i-1])/2 * (x_grid[i] - x_grid[i-1])
        
        int_to_0 = np.interp(x_eval, x_grid, cum_int)
        integral = cum_int[-1] - int_to_0
        D_approx = np.exp(-integral)
        
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return max_err

def construct():
    """
    Construct the six coefficients for the growth-index formula.
    Uses Nelder-Mead optimization to fit against numerical GR reference.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Test cosmologies (validation set - use a representative set)
    test_cosmos = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -1.0, 0.0),   # Lambda
        (0.3, -0.5, -0.5),  # CPL
        (0.3, -1.5, 0.5),   # CPL
        (0.3, -0.9, 0.3),   # CPL
        (0.2, -1.0, 0.0),   # Different Om0
        (0.3, -1.2, 0.8),   # Strong wa
    ]
    
    a_values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Initial guess: constant gamma with slight q dependence
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Nelder-Mead optimization
    best_c = c0.copy()
    best_err = _compute_error(best_c, test_cosmos, a_values)
    
    n_iter = 0
    max_iter = 200
    
    while time.time() < deadline and n_iter < max_iter:
        result = minimize(
            _compute_error,
            best_c,
            args=(test_cosmos, a_values),
            method='Nelder-Mead',
            options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        n_iter += 1
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x
        
        if time.time() > deadline - 10:
            break
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    # Final validation
    final_err = _compute_error(best_c, test_cosmos, a_values)
    if final_err > 1e3:
        # Fallback to seed
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END

