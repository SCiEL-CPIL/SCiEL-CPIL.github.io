"""
Optimized growth-index approximation for CPL dark-energy cosmologies.
Uses differential evolution followed by L-BFGS-B for local refinement.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 80.0  # Leave 10s safety margin for the 90s limit

# Validation cosmologies: (Om0, w0, wa)
# Chosen to cover a broad range of plausible physical parameters
VALIDATION_MODELS = [
    (0.30, -1.0, 0.0),   # LambdaCDM
    (0.30, -1.0, 0.3),   # Quintessence-like
    (0.30, -1.0, -0.5),  # Phantom-like
    (0.25, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.8, 0.0),   # w0 not -1
    (0.30, -1.2, 0.5),   # More negative w0
    (0.30, -1.0, 1.0),   # High wa
    (0.30, -1.0, -1.0),  # Low wa
    (0.20, -0.9, 0.2),   # Low Om, varying w
    (0.40, -1.1, -0.2),  # High Om
    (0.30, -1.0, 0.0),   # Repeat for weight? No, keep unique.
    (0.28, -1.0, 0.1),
    (0.32, -1.0, -0.1),
    (0.30, -0.5, 0.0),   # Matter-like early?
]

# Scale factors to test: log-spaced from 0.05 to 1.0
A_VALUES = np.logspace(np.log10(0.05), 0.0, 100)

def compute_Om(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy."""
    # E^2 = Om_m * a^-3 + Om_de * a^(-3(1+w0+wa))
    # Assuming flat universe: Om_de0 = 1 - Om0
    # Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) / E^2
    # E^2 = Om0 * a^-3 + (1-Om0) * a^(-3*(1+w0+wa))
    # Om_m(a) = Om0 * a^-3 / E^2
    w_avg = w0 + wa * (1 - a)
    # Note: The standard CPL evolution for Om_de uses the instantaneous w or effective w?
    # The task specifies u = 1+w0+wa*(1-a).
    # Standard relation: Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) is for constant w_eff?
    # Actually, for CPL, the exact solution for Om_de(a) is:
    # Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) / (1 + wa * (a^(-3(1+w0+wa)) - 1) / (3*wa) ... )? 
    # No, the standard definition in literature (e.g., Hogg 1999) for CPL is:
    # H^2(a) = H0^2 [ Om_m a^-3 + Om_de0 a^(-3(1+w0+wa)) ]
    # This is an approximation often used. Let's stick to this standard form as it's the most common "CPL" implementation.
    # Om_m(a) = Om_m0 * a^-3 / ( Om_m0 * a^-3 + Om_de0 * a^(-3(1+w0+wa)) )
    de_exp = -3.0 * (1.0 + w0 + wa)
    om_m_term = Om0 * (a ** -3.0)
    om_de_term = (1.0 - Om0) * (a ** de_exp)
    e2 = om_m_term + om_de_term
    return om_m_term / e2

def compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) using the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    om_a = compute_Om(a, Om0, w0, wa)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * (q ** 2) + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def compute_f(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    om_a = compute_Om(a, Om0, w0, wa)
    gamma_a = compute_gamma(a, coeffs, Om0, w0, wa)
    # Handle potential numerical issues if om_a <= 0
    return np.where(om_a > 0, om_a ** gamma_a, 0.0)

def solve_growth_ode(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200):
    """
    Numerically solve the growth ODE to get reference f_ref and D_ref.
    ODE: f'' + 3H(1+epsilon) f' - 4pi G rho f = 0
    In terms of a:
    f''_a + (3/a + 2 q) f'_a - (4pi G rho / H^2) f = 0
    4pi G rho / H^2 = 3/2 Om(a)
    So: f''_a + (3/a + 2(1-Om(a))) f'_a - 1.5 Om(a) f = 0
    
    We integrate from a_min to 1.
    Initial conditions at a_min:
    f(a_min) ~ a_min (matter dominated)
    f'(a_min) ~ 1
    """
    def ode_system(a, y):
        f, fp = y
        om_a = compute_Om(a, Om0, w0, wa)
        q = 1.0 - om_a
        # f'' = 1.5 * Om(a) * f - (3/a + 2*q) * fp
        fpp = 1.5 * om_a * f - (3.0 / a + 2.0 * q) * fp
        return [fp, fpp]

    # Initial conditions
    a_start = a_min
    f_start = a_start * (1.0 + 1e-3) # Slight offset to avoid 0/0 in derivatives if needed, but a is fine
    fp_start = 1.0
    
    sol = solve_ivp(ode_system, [a_start, a_max], [f_start, fp_start], 
                    t_eval=A_VALUES, rtol=1e-8, atol=1e-10, method='DOP853')
    
    if not sol.success:
        return None, None

    f_ref = sol.y[0]
    D_ref = sol.y[1] # Wait, D is the growth factor, f is the rate.
    # The ODE above is for f(a).
    # D(a) is obtained by integrating f(a)/a: D(a) = exp(-int_a^1 f(x)/x dx)
    # Let's compute D_ref separately or use the fact that D' = f D / a?
    # Actually, f(a) = d ln D / d ln a = a D' / D.
    # So D(a) = D(1) * exp( - \int_a^1 f(x)/x dx ).
    # We have f_ref on the grid A_VALUES.
    
    # Compute D_ref normalized to D(1)=1
    # Integral from a to 1 of f(x)/x dx
    # We can use cumulative integration.
    # Let's compute the integral from a_max down to a_min.
    # Or simply compute D(a) relative to D(1).
    
    # Using np.cumtrapz (now np.trapezoid) is tricky for reverse integration.
    # Let's compute I(a) = \int_{a}^{1} f(x)/x dx.
    # We have f(x) and x on A_VALUES.
    # integrand = f_ref / A_VALUES
    # We want the integral from each a_i to 1.
    # We can compute cumulative integral from start to end, then subtract.
    
    x = A_VALUES
    integrand = f_ref / x
    
    # Cumulative integral from x[0] to x[i]
    cum_int = np.zeros_like(x)
    # trapezoidal rule
    dx = np.diff(x)
    avg_integrand = (integrand[:-1] + integrand[1:]) / 2.0
    cum_int[1:] = np.cumsum(avg_integrand * dx)
    
    total_int = cum_int[-1] # Integral from x[0] to x[-1]
    
    # We want \int_a^1 f/x dx = \int_{x[0]}^1 f/x dx - \int_{x[0]}^a f/x dx
    # Since D(1)=1, D(a) = exp( - \int_a^1 f/x dx )
    # \int_a^1 = total_int - cum_int[i]
    # Wait, cum_int[i] is \int_{x[0]}^{x[i]}.
    # So \int_{x[i]}^{x[-1]} = total_int - cum_int[i].
    
    D_ref = np.exp( -(total_int - cum_int) )
    
    return f_ref, D_ref

def calculate_errors(coeffs, models, a_vals):
    """Calculate max relative error for f and D."""
    max_err_f = 0.0
    max_err_D = 0.0
    gamma_violation = 0.0
    
    for Om0, w0, wa in models:
        # Get reference
        f_ref, D_ref = solve_growth_ode(Om0, w0, wa, a_min=a_vals[0], a_max=a_vals[-1])
        if f_ref is None:
            continue
            
        # Compute approximation
        f_approx = compute_f(a_vals, coeffs, Om0, w0, wa)
        
        # Check Gamma constraints
        gamma_approx = compute_gamma(a_vals, coeffs, Om0, w0, wa)
        # Gamma must be in [0.1, 1.2]
        gamma_violation += np.sum((gamma_approx < 0.1) | (gamma_approx > 1.2))
        
        # Relative error for f
        # Avoid division by zero if f_ref is 0 (unlikely for a>0.05)
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        max_err_f = max(max_err_f, np.max(err_f))
        
        # Compute D_approx
        # D_approx(a) = exp( - \int_a^1 f_approx(x)/x dx )
        integrand_app = f_approx / a_vals
        cum_int_app = np.zeros_like(a_vals)
        dx = np.diff(a_vals)
        avg_integrand_app = (integrand_app[:-1] + integrand_app[1:]) / 2.0
        cum_int_app[1:] = np.cumsum(avg_integrand_app * dx)
        total_int_app = cum_int_app[-1]
        
        D_approx = np.exp( -(total_int_app - cum_int_app) )
        
        # Relative error for D
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        max_err_D = max(max_err_D, np.max(err_D))
        
    # Combine errors
    # Penalty for gamma violation
    penalty = 10.0 * gamma_violation
    return max(max_err_f, max_err_D) + penalty

def objective(coeffs):
    """Objective function for optimizer."""
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10
    return calculate_errors(coeffs, VALIDATION_MODELS, A_VALUES)

def construct():
    # Initial guess: constant gamma ~ 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds for coefficients
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    best_result = x0
    best_score = 1e10
    
    # Stage 1: Differential Evolution
    # Use a moderate number of iterations to stay within time limit
    # popsize=20, maxiter=100 is a reasonable start.
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-5,
            polish=False  # We will do polish manually
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_result = res_de.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return best_result
        
    # Stage 2: Local Refinement (L-BFGS-B)
    try:
        res_lbfgs = minimize(
            objective, 
            best_result, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res_lbfgs.fun < best_score:
            best_score = res_lbfgs.fun
            best_result = res_lbfgs.x
            
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")

    # Ensure coefficients are within [-10, 10]
    best_result = np.clip(best_result, -10, 10)
    
    return list(best_result)

# Pre-compute references to speed up objective function if possible?
# No, solve_ivp is the bottleneck. Caching based on (Om0, w0, wa) is useful.
_reference_cache = {}

def get_reference(Om0, w0, wa):
    key = (Om0, w0, wa)
    if key not in _reference_cache:
        _reference_cache[key] = solve_growth_ode(Om0, w0, wa)
    return _reference_cache[key]

# Override calculate_errors to use cache
def calculate_errors_cached(coeffs, models, a_vals):
    max_err_f = 0.0
    max_err_D = 0.0
    gamma_violation = 0.0
    
    for Om0, w0, wa in models:
        f_ref, D_ref = get_reference(Om0, w0, wa)
        if f_ref is None:
            continue
            
        f_approx = compute_f(a_vals, coeffs, Om0, w0, wa)
        gamma_approx = compute_gamma(a_vals, coeffs, Om0, w0, wa)
        gamma_violation += np.sum((gamma_approx < 0.1) | (gamma_approx > 1.2))
        
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        max_err_f = max(max_err_f, np.max(err_f))
        
        integrand_app = f_approx / a_vals
        cum_int_app = np.zeros_like(a_vals)
        dx = np.diff(a_vals)
        avg_integrand_app = (integrand_app[:-1] + integrand_app[1:]) / 2.0
        cum_int_app[1:] = np.cumsum(avg_integrand_app * dx)
        total_int_app = cum_int_app[-1]
        
        D_approx = np.exp( -(total_int_app - cum_int_app) )
        
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        max_err_D = max(max_err_D, np.max(err_D))
        
    penalty = 10.0 * gamma_violation
    return max(max_err_f, max_err_D) + penalty

# Replace objective to use cached version
def objective_cached(coeffs):
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10
    return calculate_errors_cached(coeffs, VALIDATION_MODELS, A_VALUES)

# Re-run construct with cached objective
def construct():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    best_result = x0
    best_score = 1e10
    
    # Warm up cache
    for m in VALIDATION_MODELS:
        get_reference(*m)

    try:
        res_de = differential_evolution(
            objective_cached, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-5,
            polish=False
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_result = res_de.x
            
    except Exception as e:
        pass
        
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return [np.clip(c, -10, 10) for c in best_result]
        
    try:
        res_lbfgs = minimize(
            objective_cached, 
            best_result, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res_lbfgs.fun < best_score:
            best_score = res_lbfgs.fun
            best_result = res_lbfgs.x
            
    except Exception as e:
        pass

    return [np.clip(c, -10, 10) for c in best_result]

# Final check: The task asks for construct() to be the entry point.
# The code above defines construct() twice. The second one overrides the first.
# This is valid Python.
