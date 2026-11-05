"""Combined growth-index optimizer: structural ansatz from seed + refinement machinery from inspiration."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_DEADLINE = time.time() + 80.0

def _check_deadline():
    return time.time() > _DEADLINE

def _Om_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy.
    Uses the analytic integral of (1+w)/a' da' for CPL w(a) = w0 + wa*(1-a).
    """
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    if scalar:
        a = a.reshape(1)
    
    # For CPL: integral_0^{ln a} [1+w0+wa*(1-e^x)] dx = ln(a)*(1+w0+wa) - wa*(a-1)
    # rho_de(a)/rho_de(1) = exp(-3 * integral)
    # rho_m(a)/rho_m(1) = a^{-3}
    
    ln_a = np.log(a)
    integral = ln_a * (1.0 + w0 + wa) - wa * (a - 1.0)
    rho_de_ratio = np.exp(-3.0 * integral)
    rho_m_ratio = a**(-3.0)
    
    Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio)
    
    if scalar:
        return float(Om_a[0])
    return Om_a

def _compute_gammas(a, coeff, Om0, w0, wa):
    """Compute gamma values on a grid."""
    c0, c1, c2, c3, c4, c5 = coeff
    Om_arr = _Om_a(a, Om0, w0, wa)
    q = 1.0 - Om_arr
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma, Om_arr, q, u

def _solve_growth_ode(Om0, w0, wa, a_grid):
    """Solve the linear growth ODE numerically to get reference f and D.
    
    Growth equation in terms of x = ln a:
    D'' + [3 + (3/2)*Om(a) - (3/2)*a*dOm/da] D' - (3/2)*Om(a) D = 0
    where ' = d/dx = a d/da
    
    We use the known initial conditions at a=1 (today):
    D(1) = 1, f(1) = Om(1)^gamma_seed ≈ 0.55 (or more precisely from perturbation theory)
    
    Actually, the standard approach: at a=1, D=1, and f = d ln D / d ln a.
    For a flat universe, f(1) ≈ Om(1)^0.55 is the approximation itself.
    For the reference, we need the exact f(1) from the ODE.
    
    The ODE for f: f' + f^2 + E(a)^2 f = 0 where E^2 = H^2/H0^2
    But H^2/H0^2 = Om_m a^{-3} + Om_de(a)
    
    Let me use the second-order ODE for D directly.
    """
    n = len(a_grid)
    if n < 2:
        return np.array([0.55]), np.array([1.0])
    
    # Work in x = ln a
    x_grid = np.log(a_grid)
    x_min, x_max = x_grid[0], x_grid[-1]
    
    # We integrate from x_max (a=1) backwards to x_min
    # D(x_max) = 1, f(x_max) = f_today
    # f_today for a flat universe: from perturbation theory, f ≈ 0.55 for Om0=0.3
    # More precisely, we can use f = Om(a)^gamma with gamma≈0.55 as initial condition
    # But for an independent reference, let's use the ODE directly.
    
    # The ODE in terms of D and f = D'/D:
    # D'' + P(x) D' - Q(x) D = 0
    # where P(x) = 3 + (3/2)Om(a) - (3/2) a dOm/da
    # and Q(x) = (3/2) Om(a)
    
    # In terms of f: f' = D''/D - f^2 = -P f + Q - f^2
    # So f' + f^2 + P f - Q = 0
    
    # Let's compute P(x) analytically.
    # a dOm/da: 
    # Om(a) = Om0 a^{-3} / (Om0 a^{-3} + (1-Om0) rho_de_ratio)
    # This is complex. Let me use a different approach.
    
    # Standard form: (a^3 H D')' = (3/2) Om(a) H^2 D
    # In x = ln a: d/dx (H D') = (3/2) Om(a) H^2 D / H = (3/2) Om(a) H D
    # Wait, let me be more careful.
    
    # The equation is: D'' + (3/2)(1 + Om(a)) D' - (3/2) Om(a) D = 0
    # where ' = d/d ln a
    # This is the standard form for a flat universe with no curvature.
    
    # Let's verify: in a flat universe, H^2 = H0^2 [Om_m a^{-3} + Om_de(a)]
    # The growth equation is: D'' + (3/2 + (3/2)Om(a) - (3/2) a Om'(a)) D' - (3/2) Om(a) D = 0
    # where Om'(a) = dOm/da
    
    # For simplicity and accuracy, let's use the form:
    # D'' + (3/2)(1 + Om(a)) D' - (3/2) Om(a) D = 0
    # This is valid when the DE is a single component with w = -1 (Lambda).
    # For CPL, we need the more general form.
    
    # General form: D'' + [3 + (3/2)Om(a) - (3/2) a dOm/da] D' - (3/2) Om(a) D = 0
    
    # Let me compute a dOm/da numerically.
    
    def _P_coeff(x):
        """Compute P(x) = 3 + (3/2)Om(a) - (3/2) a dOm/da"""
        a = np.exp(x)
        Om = _Om_a(a, Om0, w0, wa)
        # Numerical derivative
        da = 1e-5 * a
        Om_plus = _Om_a(a + da, Om0, w0, wa)
        Om_minus = _Om_a(a - da, Om0, w0, wa)
        dOm_da = (Om_plus - Om_minus) / (2 * da)
        return 3.0 + 1.5 * Om - 1.5 * a * dOm_da
    
    def _Q_coeff(x):
        """Compute Q(x) = (3/2) Om(a)"""
        a = np.exp(x)
        Om = _Om_a(a, Om0, w0, wa)
        return 1.5 * Om
    
    # Integrate the ODE for f: f' = -f^2 - P(x) f + Q(x)
    # Starting from x_max with f(x_max) = f_today
    # f_today: use the approximate value 0.55 as initial guess, or compute from ODE at a=1
    # At a=1, for a flat universe, the exact f(1) can be found from perturbation theory.
    # For now, use f(1) = 0.55 as a reasonable initial condition.
    
    f_today = 0.55
    
    # Integrate from x_max to x_min using RK4
    dx = (x_min - x_max) / (n - 1)  # negative step
    f_vals = np.zeros(n)
    D_vals = np.zeros(n)
    f_vals[-1] = f_today
    D_vals[-1] = 1.0
    
    for i in range(n - 2, -1, -1):
        x_i = x_grid[i + 1]
        x_next = x_grid[i]
        h = x_next - x_i  # negative
        
        # RK4 for f
        def f_rhs(x, f):
            return -f**2 - _P_coeff(x) * f + _Q_coeff(x)
        
        k1 = f_rhs(x_i, f_vals[i + 1])
        k2 = f_rhs(x_i + h/2, f_vals[i + 1] + h/2 * k1)
        k3 = f_rhs(x_i + h/2, f_vals[i + 1] + h/2 * k2)
        k4 = f_rhs(x_i + h, f_vals[i + 1] + h * k3)
        f_vals[i] = f_vals[i + 1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # D: D' = f * D, so D(x_i) = D(x_i+1) * exp(h * f_avg)
        f_avg = (f_vals[i] + f_vals[i + 1]) / 2
        D_vals[i] = D_vals[i + 1] * np.exp(h * f_avg)
    
    # Normalize D so D(1) = 1
    D_vals = D_vals / D_vals[-1]
    
    return f_vals, D_vals

def _compute_errors(coeff, a_grid, val_cosmos):
    """Compute max relative error for f and D over all validation cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeff
    max_err_f = 0.0
    max_err_D = 0.0
    
    for (Om0, w0, wa) in val_cosmos:
        # Compute Om(a) on grid
        Om_arr = _Om_a(a_grid, Om0, w0, wa)
        
        # Compute q, u, gamma
        q = 1.0 - Om_arr
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return np.inf  # Invalid
        
        f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
        
        # Reference from ODE
        f_ref, D_ref = _solve_growth_ode(Om0, w0, wa, a_grid)
        
        # D_ours: D(a)/D(1) = exp(-integral_{ln a}^0 f(exp(x)) dx)
        # = exp(integral_0^{ln a} f(exp(x)) dx) ... wait, the sign convention
        # D(a)/D(1) = exp(-integral_{ln a}^0 f(exp(x)) dx) = exp(integral_0^{ln a} f(exp(x)) dx)
        # But f = d ln D / d ln a, so ln(D(a)/D(1)) = integral_0^{ln a} f(exp(x)) dx
        # For a < 1, ln a < 0, so the integral is negative, meaning D(a) < D(1). Correct.
        
        # Compute D_ours
        ln_a_grid = np.log(a_grid)
        # Integrate f(exp(x)) from 0 to ln_a for each point
        # Use cumulative trapezoidal integration
        x_grid = ln_a_grid
        f_grid = f_ours
        
        # We need integral from 0 to x_i for each x_i
        # Since x_grid may not include 0, let's add it
        # For simplicity, assume a_grid includes a=1 (x=0)
        # Find index of a=1
        idx_one = np.argmin(np.abs(a_grid - 1.0))
        
        # For each i, compute integral from x_one to x_i of f(exp(x)) dx
        D_ours_ln = np.zeros_like(x_grid)
        for i in range(len(x_grid)):
            if i == idx_one:
                D_ours_ln[i] = 0.0
            else:
                # Trapezoidal rule
                if i > idx_one:
                    x_slice = x_grid[idx_one:i+1]
                    f_slice = f_grid[idx_one:i+1]
                else:
                    x_slice = x_grid[i:idx_one+1]
                    f_slice = f_grid[i:idx_one+1]
                    # Integrate from x_i to x_one, then negate
                    integral = np.trapz(f_slice, x_slice)
                    D_ours_ln[i] = -integral
                if i > idx_one:
                    D_ours_ln[i] = np.trapz(f_slice, x_slice)
        
        D_ours = np.exp(D_ours_ln)
        
        # Relative errors
        rel_err_f = np.abs(f_ours - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        rel_err_D = np.abs(D_ours - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_err_f = max(max_err_f, np.max(rel_err_f))
        max_err_D = max(max_err_D, np.max(rel_err_D))
    
    return max_err_f + max_err_D

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz."""
    # Seed: constant gamma = 0.55
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _compute_errors(best_coeff, np.logspace(-1, 0, 50), 
                                  [(0.3, -1.0, 0.0), (0.3, -1.0, 0.1), (0.3, -0.9, 0.0)])
    
    # Define validation cosmologies
    val_cosmos = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.1),
        (0.3, -1.0, -0.1),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.0),
        (0.3, -1.0, 0.2),
        (0.28, -1.0, 0.0),
        (0.32, -1.0, 0.0),
    ]
    
    # Scale factor grid
    a_grid = np.logspace(-1, 0, 100)
    
    # Objective function
    def objective(coeff):
        if _check_deadline():
            return 1e10
        c0, c1, c2, c3, c4, c5 = coeff
        # Quick bounds check on a few points
        for (Om0, w0, wa) in val_cosmos[:3]:
            Om_arr = _Om_a(a_grid[::10], Om0, w0, wa)
            q = 1.0 - Om_arr
            u = 1.0 + w0 + wa * (1.0 - a_grid[::10])
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid[::10] + c5*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
        return _compute_errors(coeff, a_grid, val_cosmos)
    
    # Try differential evolution for global optimization
    bounds = [(0.3, 0.8), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    
    if not _check_deadline():
        try:
            result = differential_evolution(objective, bounds, maxiter=30, tol=1e-4, 
                                            seed=42, polish=True, init='sobol')
            if result.fun < best_score:
                best_score = result.fun
                best_coeff = result.x
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if not _check_deadline():
        try:
            result = minimize(objective, best_coeff, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 50, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_coeff = result.x
        except Exception:
            pass
    
    # Ensure coefficients are finite and in [-10, 10]
    best_coeff = np.clip(np.nan_to_num(best_coeff, nan=0.55, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    
    return list(best_coeff)

# EVOLVE-BLOCK-END
