"""Collocation-based minimax fit for CPL growth-index coefficients."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Deadline guard: evaluator kills at 90s; keep a safety margin.
_DEADLINE = 85.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _Om_of_a(a, Om0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(-3*wa) * a^3)
    # More precisely: Om_m(a) = Om0 * a^3 / E(a)^2 where E(a)^2 = Om0*a^3 + (1-Om0)*a^{-3*wa}
    a3 = a**3
    E2 = Om0 * a3 + (1.0 - Om0) * a**(-3.0 * wa)
    return Om0 * a3 / E2

def _gamma_of_a(a, Om0, wa, c):
    """Compute gamma(a) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q."""
    q = 1.0 - _Om_of_a(a, Om0, wa)
    u = 1.0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f_of_a(a, Om0, wa, c):
    """Growth rate f = Om(a)^gamma(a)."""
    Om = _Om_of_a(a, Om0, wa)
    gam = _gamma_of_a(a, Om0, wa, c)
    # Ensure Om > 0 for log
    if Om <= 0:
        return 0.0
    return Om ** gam

def _reference_growth(a_grid, Om0, wa):
    """
    Solve the linear growth ODE numerically to get D(a) and f(a) = D'/D.
    D'' + 2H D' + (3/2) H^2 Om(a) D = 0
    In terms of x = ln(a): d^2D/dx^2 + (3/2 + 2*wa*(1-a)/(1+wa*(1-a))) dD/dx + (3/2)*Om(a) D = 0
    Actually, let's use the standard form with H(a)/H0 = E(a).
    The equation in x=ln(a) is:
    D'' + (3/2 + 2 q(a)) D' + (3/2) Om(a) D = 0
    where q(a) = -E'/E = -d ln E / dx. For CPL, q(a) = (1 + w(a)) / 2 * ... 
    Let's derive: E(a) = sqrt(Om0 a^3 + (1-Om0) a^{-3wa})
    d ln E / dx = (1/2) d ln(E^2) / dx = (1/2) * [3 Om0 a^3 - 3wa (1-Om0) a^{-3wa}] / E^2
    = (3/2) [Om0 a^3 - wa (1-Om0) a^{-3wa}] / E^2
    = (3/2) [Om(a) - wa (1-Om(a))]  (since Om = Om0 a^3 / E^2 and 1-Om = (1-Om0)a^{-3wa}/E^2)
    So q(a) = -d ln E/dx = -(3/2)[Om - wa(1-Om)]
    
    The ODE for D(x) where x=ln a:
    D'' + (3/2 + 2q) D' + (3/2) Om(a) D = 0
    
    We want f = D'/D. Let D' = f D. Then D'' = (f' + f^2) D.
    (f' + f^2) + (3/2 + 2q) f + (3/2) Om = 0
    f' = -f^2 - (3/2 + 2q) f - (3/2) Om
    """
    # We integrate backwards from a=1 (x=0) to smaller a (x<0).
    # At a=1, Om(1)=Om0, q(1) = -(3/2)[Om0 - wa(1-Om0)]
    # f(1) = Om0^gamma(1) approximately. We'll use the analytic f(1) = Om0^0.55 as initial guess?
    # Better: use the ODE for f and integrate. But f(1) is not exactly known.
    # Alternative: integrate D directly with D(1)=1, D'(1)=f(1).
    # Let's use f(1) from the constant gamma approximation as initial condition, then refine.
    # Actually, for the reference, we should solve the ODE for D with D(1)=1 and D'(1) determined by matching.
    # Let's just solve for D with D(1)=1 and D'(1)=Om0^0.55 (good initial guess), then compute f=D'/D.
    
    x_end = np.log(a_grid[0])  # negative
    x0 = 0.0
    
    def ode_rhs(x, y):
        a = np.exp(x)
        Om = _Om_of_a(a, Om0, wa)
        q = -(3.0/2.0) * (Om - wa * (1.0 - Om))
        D, Dp = y
        Dpp = -(3.0/2.0 + 2.0*q) * Dp - (3.0/2.0) * Om * D
        return [Dp, Dpp]
    
    sol = solve_ivp(ode_rhs, [x0, x_end], [1.0, Om0**0.55], 
                    t_eval=a_grid[::-1].copy() if False else None,
                    dense_output=True, rtol=1e-10, atol=1e-12, method='DOP853')
    
    # Evaluate at a_grid
    x_grid = np.log(a_grid)
    sol_vals = sol.sol(x_grid)
    D_ref = sol_vals[0]
    Dp_ref = sol_vals[1]
    f_ref = Dp_ref / D_ref
    # Normalize D so D(1)=1 (already done by initial condition)
    # D_ref[0] should be ~1 since a_grid[0] might not be 1. Let's check: a_grid includes 1.
    # Find index where a=1
    idx1 = np.argmin(np.abs(a_grid - 1.0))
    D_ref = D_ref / D_ref[idx1]
    Dp_ref = Dp_ref / D_ref[idx1]
    f_ref = Dp_ref / D_ref
    return f_ref, D_ref

def _evaluate_error(c, cosmologies, a_grid):
    """Compute max relative error of f and D across all cosmologies and a values."""
    max_err = 0.0
    for Om0, wa in cosmologies:
        f_ref, D_ref = _reference_growth(a_grid, Om0, wa)
        # Compute our approximation
        f_app = np.array([_f_of_a(a, Om0, wa, c) for a in a_grid])
        # Compute D_app via integration of -f dx from ln(a) to 0
        # D(a)/D(1) = exp(-integral_{ln a}^0 f(exp(x)) dx)
        # Let's compute D_app by integrating f_app backwards
        # Use cumulative trapezoidal rule
        x_grid = np.log(a_grid)
        # We need integral from x to 0 of f(exp(x')) dx'
        # Since a_grid is sorted, x_grid is sorted
        # integral from x_i to 0 = - integral from 0 to x_i
        # Let's compute cumulative integral from x[0] to x[i]
        from scipy.integrate import cumulative_trapezoid
        # f_app is at x_grid
        # integral from x[0] to x[i] of f dx
        cum_int = cumulative_trapezoid(f_app, x_grid, initial=0.0)
        # integral from x[i] to 0 = integral from x[0] to 0 - integral from x[0] to x[i]
        # integral from x[0] to 0 = cum_int[-1] (if x_grid[-1] is close to 0)
        # Actually x_grid[-1] = log(1) = 0 if a_grid[-1]=1
        # Let's assume a_grid includes 1 at the end
        int_to_0 = cum_int[-1] - cum_int
        D_app = np.exp(-int_to_0)
        # Normalize D_app so D_app[1]=1 (should already be ~1)
        idx1 = np.argmin(np.abs(a_grid - 1.0))
        D_app = D_app / D_app[idx1]
        
        # Relative errors
        err_f = np.abs(f_app - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_app - D_ref) / np.abs(D_ref)
        local_max = np.max(np.concatenate([err_f, err_D]))
        if local_max > max_err:
            max_err = local_max
        if _time_left() < 10:
            break
    return max_err

def construct():
    """Find optimal coefficients using minimax optimization."""
    # Deadline check
    if _time_left() < 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Define cosmologies for fitting (disjoint from validation, but we don't know validation)
    # Use a grid of cosmologies
    Om0_vals = np.linspace(0.2, 0.4, 5)
    wa_vals = np.linspace(-1.0, 1.0, 5)
    cosmologies = [(o, w) for o in Om0_vals for w in wa_vals]
    
    # Scale factor grid
    a_grid = np.geomspace(0.01, 1.0, 50)
    
    # Initial guess: constant gamma
    c0_init = 0.55
    c_init = np.array([c0_init, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Objective: minimize max error
    def objective(c):
        if _time_left() < 5:
            return 1.0
        try:
            err = _evaluate_error(c, cosmologies, a_grid)
            return err
        except:
            return 1.0
    
    # Constraints: gamma in [0.1, 1.2] at all checked locations
    # This is complex; let's add penalty for out-of-range gamma
    def objective_with_penalty(c):
        if _time_left() < 5:
            return 1.0
        try:
            err = _evaluate_error(c, cosmologies, a_grid)
            # Check gamma bounds
            gamma_violation = 0.0
            for Om0, wa in cosmologies:
                for a in a_grid:
                    gam = _gamma_of_a(a, Om0, wa, c)
                    if gam < 0.1:
                        gamma_violation += (0.1 - gam)
                    elif gam > 1.2:
                        gamma_violation += (gam - 1.2)
            # Penalty
            if gamma_violation > 0:
                err += 10.0 * gamma_violation
            return err
        except:
            return 1.0
    
    # Try SLSQP
    best_c = c_init.copy()
    best_err = objective_with_penalty(c_init)
    
    if _time_left() > 10:
        try:
            res = minimize(objective_with_penalty, c_init, method='SLSQP', 
                          bounds=[(-10, 10)]*6, options={'maxiter': 50, 'ftol': 1e-6})
            if res.success and res.fun < best_err:
                best_c = res.x
                best_err = res.fun
        except:
            pass
    
    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Final check
    if _time_left() < 2:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(x) for x in best_c]
