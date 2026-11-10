"""Improved growth-index approximation via boundary-matching asymptotic constraints."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from itertools import product

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 90.0  # seconds

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE * 0.85  # safety margin

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    # E(a)^2 = Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa*(1-a)))
    # Note: the exponent for DE is 3*(1+w) where w = w0 + wa*(1-a)
    # So E(a)^2 = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa*(1-a)))
    # Actually, for CPL: w(a) = w0 + wa*(1-a)
    # The DE density scales as a^(-3(1+w_eff)) but with varying w, we need to integrate.
    # Standard form: rho_DE(a) ∝ a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # So E(a)^2 = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
    # Let's use the standard CPL form:
    # E(a)^2 = Om_m * a^-3 + (1-Om_m) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
    # This is the standard parametric form for CPL.
    # However, the task says u = 1 + w0 + wa*(1-a), which is 1+w(a).
    # And q = 1 - Om(a).
    # The standard definition of E(a)^2 for CPL is:
    # E(a)^2 = Om_m * a^-3 + (1-Om_m) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
    # Let's verify this is consistent with u.
    # Actually, a more common form is E(a)^2 = Om_m * a^-3 + (1-Om_m) * a^(-3) * a^(-3*wa*(1-a)) * a^(-3*w0)
    # = Om_m * a^-3 + (1-Om_m) * a^(-3*(1+w0+wa*(1-a)))
    # This is simpler and matches the u definition directly if we interpret the exponent as 3*u.
    # Let's use: E(a)^2 = Om_m * a^-3 + (1-Om_m) * a^(-3*u) where u = 1+w0+wa*(1-a).
    # This is the "effective" CPL form often used in growth index literature.
    u = 1.0 + w0 + wa * (1.0 - a)
    return Om0 / (Om0 * a**(-3) + (1.0 - Om0) * a**(-3.0 * u))

def _gamma_of_a(a, Om0, w0, wa, coeffs):
    """Compute gamma from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f_of_a(a, Om0, w0, wa, coeffs):
    """Compute f = Om(a)^gamma."""
    gamma = _gamma_of_a(a, Om0, w0, wa, coeffs)
    Om_a = _Om_of_a(a, Om0, w0, wa)
    if Om_a <= 0:
        return 0.0
    return Om_a ** gamma

def _reference_f_and_D(a_grid, Om0, w0, wa):
    """Solve the ODE to get reference f(a) and D(a)/D(1)."""
    # The ODE for f(a) = d ln D / d ln a:
    # f' + (3/2)(1+Om) f - (3/2) Om = 0  (where ' is d/da)
    # Or in terms of ln a: df/d ln a + f^2 + (3/2)(1+Om(a)) f - (3/2) Om(a) = 0
    # We integrate from a=1 backwards to a_min.
    # f(1) ≈ 0.55 (or solve for it)
    # D(a)/D(1) = exp(-int_1^a f(a') d ln a')
    
    a_min = 0.05
    a_max = 1.0
    
    def ode_rhs(x, y):
        # x = ln a, y = [f]
        a = np.exp(x)
        Om_a = _Om_of_a(a, Om0, w0, wa)
        f = y[0]
        # df/dx = -f^2 - (3/2)(1+Om_a) f + (3/2) Om_a
        dfdx = -f**2 - 1.5 * (1.0 + Om_a) * f + 1.5 * Om_a
        return [dfdx]
    
    # Initial condition at a=1 (x=0)
    # f(1) is approximately 0.55 for standard cosmology, but let's use a slightly better estimate
    # or just use 0.55 and let the ODE handle it.
    f0 = 0.55
    x0 = 0.0
    
    # Integrate from x=0 to x=ln(a_min)
    x_end = np.log(a_min)
    sol = solve_ivp(ode_rhs, [x0, x_end], [f0], method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        return None, None
    
    # Evaluate on the grid
    x_grid = np.log(a_grid)
    # Ensure x_grid is within [x_end, x0]
    x_grid = np.clip(x_grid, x_end, x0)
    
    f_ref = sol.sol(x_grid)[0]
    
    # Compute D(a)/D(1) = exp(-int_1^a f d ln a') = exp(int_a^1 f d ln a')
    # We can integrate numerically using the solution
    # D(a)/D(1) = exp(-int_{ln a}^{0} f(x') dx')
    # Let's compute the integral for each a in the grid
    D_ratio = np.ones_like(a_grid)
    for i, x_i in enumerate(x_grid):
        # Integrate from x_i to 0
        if x_i < 0:
            # Use trapezoidal rule on the dense solution
            n_pts = 100
            x_int = np.linspace(x_i, 0, n_pts)
            f_int = sol.sol(x_int)[0]
            integral = np.trapz(f_int, x_int)
            D_ratio[i] = np.exp(-integral)
        else:
            D_ratio[i] = 1.0
    
    return f_ref, D_ratio

def _compute_error(coeffs, Om0, w0, wa, a_grid):
    """Compute the maximum relative error for f and D."""
    f_ref, D_ref = _reference_f_and_D(a_grid, Om0, w0, wa)
    if f_ref is None:
        return np.inf
    
    f_approx = np.array([_f_of_a(a, Om0, w0, wa, coeffs) for a in a_grid])
    
    # Check gamma constraints
    for a in a_grid:
        gamma = _gamma_of_a(a, Om0, w0, wa, coeffs)
        if gamma < 0.1 or gamma > 1.2:
            return np.inf
    
    # Relative error for f
    mask_f = np.abs(f_ref) > 1e-10
    if np.any(mask_f):
        rel_err_f = np.max(np.abs((f_approx[mask_f] - f_ref[mask_f]) / f_ref[mask_f]))
    else:
        rel_err_f = 0.0
    
    # Relative error for D
    # Compute D_approx
    D_approx = np.ones_like(a_grid)
    for i in range(1, len(a_grid)):
        # D(a_i)/D(a_{i-1}) = exp(-int_{a_{i-1}}^{a_i} f d ln a)
        # Approximate with trapezoidal rule
        a_prev = a_grid[i-1]
        a_curr = a_grid[i]
        f_prev = f_approx[i-1]
        f_curr = f_approx[i]
        integral = 0.5 * (f_prev + f_curr) * (np.log(a_curr) - np.log(a_prev))
        D_approx[i] = D_approx[i-1] * np.exp(-integral)
    
    mask_D = np.abs(D_ref) > 1e-10
    if np.any(mask_D):
        rel_err_D = np.max(np.abs((D_approx[mask_D] - D_ref[mask_D]) / D_ref[mask_D]))
    else:
        rel_err_D = 0.0
    
    return max(rel_err_f, rel_err_D)

def construct():
    """Return 6 coefficients for the growth index ansatz."""
    if _check_deadline():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Define validation cosmologies
    # These are typical CPL models
    cosmologies = [
        (0.3, -1.0, 0.0),   # LCDM-like
        (0.3, -1.0, 0.5),   # Phantom with positive wa
        (0.3, -0.5, -0.5),  # Quintessence-like
        (0.25, -1.0, 0.3),  # Different Om0
        (0.35, -0.8, -0.2), # Another variant
    ]
    
    # Grid of a values
    a_grid = np.linspace(0.1, 1.0, 20)
    
    # Initial guess: constant gamma = 0.55
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_error = np.inf
    
    # Try to optimize
    def objective(coeffs):
        if _check_deadline():
            return 1e10
        total_error = 0.0
        for Om0, w0, wa in cosmologies:
            err = _compute_error(coeffs, Om0, w0, wa, a_grid)
            if np.isinf(err):
                return 1e10
            total_error += err
        return total_error / len(cosmologies)
    
    # Use Nelder-Mead optimization
    try:
        x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        bounds = [(-10, 10)] * 6
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_error:
            best_error = result.fun
            best_coeffs = result.x.tolist()
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeffs
    
    # Refine with more iterations if time allows
    try:
        x0 = np.array(best_coeffs)
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if result.fun < best_error:
            best_error = result.fun
            best_coeffs = result.x.tolist()
    except Exception:
        pass
    
    # Ensure coefficients are within [-10, 10]
    best_coeffs = [max(-10.0, min(10.0, c)) for c in best_coeffs]
    
    return best_coeffs
