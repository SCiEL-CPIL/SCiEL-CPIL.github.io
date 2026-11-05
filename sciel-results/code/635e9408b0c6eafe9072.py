"""Growth-index approximation for CPL dark-energy cosmologies.

This module performs a bounded, deadline-guarded local search at import time to
optimize six coefficients for the compact growth-index ansatz.  The evaluator
calls construct(), which simply returns the pre-computed coefficient list.

Public task contract:
    f = Omega_m(a) ** gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)

    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    q = 1 - Omega_m(a)
    u = 1 + w0 + wa*(1 - a)

    gamma must lie in [0.1, 1.2] at all checked locations.
    Coefficients must be in [-10, 10].

Score = 100 * max relative error over both f and normalized D
on disjoint validation cosmologies.
"""

import numpy as np
from scipy.optimize import minimize
import time

# ── Deadline guard ──────────────────────────────────────────────────────────
_START_TIME = time.time()
_TIME_LIMIT = 85.0   # leave 5 s safety margin on a 90 s wall clock

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ─────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor.

    Om_m(a) = Om0 * a^{-3} / [Om0*a^{-3} + (1-Om0)*a^{-3(1+w0+wa)}*exp(3*wa*(a-1))]
    """
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    # Dark energy factor for CPL
    # rho_DE(a) ~ a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    Om_m = Om0 * a_inv3 / (Om0 * a_inv3 + Om_DE)
    return Om_m

def _gamma_from_coeffs(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_from_gamma(gamma, Om_m):
    """f = Om_m^gamma."""
    Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
    return Om_m_safe ** gamma

# ── Reference: solve the growth ODE numerically ────────────────────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """Compute f(a) and D(a)/D(1) by solving the linear growth ODE.

    We integrate the second-order ODE forward from a small a to a=1.
    """
    from scipy.integrate import solve_ivp

    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        # dOm_m/dx = -3 Om_m (Om_m - 1)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        # Growth equation: D'' + [3/2 + (1+w) - 3/2 dOm_m/dx] D' - 3/2 Om_m D = 0
        D, F = y
        D_prime = F
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [D_prime, D_double]

    # Initial conditions at a_min: matter era approximation
    # D ~ a^(5/6), F = dD/dx = (5/6) a^(-1/6)
    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.005, rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        
        # Normalize D(1)=1
        D_ref = D_ref / D_ref[-1]
        
        return F_ref, D_ref

    except Exception:
        # Fallback: constant gamma reference
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        
        # Compute D via integration
        x = np.log(a_grid)
        # integral from 0 to x of f(e^t) dt
        # Use cumulative trapezoid
        # We want integral from 0 to x_i. Since x is increasing, we can integrate from start and subtract?
        # Easier: integrate from end (x=0) backwards?
        # Let's use the same logic as the model calculation for consistency in fallback
        # But for reference, we just need some valid D.
        # D(a)/D(1) = exp( \int_0^{ln a} f(e^t) dt )
        # Let's compute cumulative integral of f w.r.t x from x_min to x_max.
        # I(x) = \int_{x_min}^x f(t) dt
        # \int_0^x f(t) dt = I(x) - I(0)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        # I(x_i) is cum_int[i]
        # We need \int_0^{x_i} f(t) dt.
        # Note x_max = 0.
        # I(x_i) = \int_{x_min}^{x_i} f(t) dt
        # I(x_max) = \int_{x_min}^{0} f(t) dt
        # \int_0^{x_i} f(t) dt = \int_{x_min}^{x_i} f(t) dt - \int_{x_min}^{0} f(t) dt
        # = cum_int[i] - cum_int[-1]
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        
        return f_ref, D_ref

# ── Objective function ──────────────────────────────────────────────────────

# Validation cosmologies (typical CPL range)
_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),   # wCDM
    (0.30, -1.0, 0.0),   # LambdaCDM
    (0.30, -1.0, 0.5),   # CPL positive wa
    (0.30, -1.0, -0.5),  # CPL negative wa
    (0.30, -0.8, 0.3),   # intermediate
    (0.25, -1.0, 0.5),   # low Om, positive wa
    (0.35, -1.0, -0.3),  # high Om, negative wa
    (0.30, -1.2, 0.4),   # phantom-like
    (0.30, -0.9, -0.4),  # quintessence-like
    (0.28, -1.0, 1.0),   # extreme wa
]

# Fixed evaluation grid
_EVAL_A = np.logspace(np.log10(1e-3), 0.0, 200)

# Pre-compute reference solutions on the SAME grid as evaluation
_REF_DATA = []
def _build_references():
    """Pre-compute reference growth data for all validation cosmologies."""
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa))
        except Exception:
            _REF_DATA.append(None)

def _compute_errors_for_coeffs(c):
    """Compute all relative errors for a given coefficient vector.
    Returns (max_err, list_of_all_relative_errors)."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6, []
    if np.any(c < -10) or np.any(c > 10):
        return 1e6, []

    all_errors = []
    a_eval = _EVAL_A
    x_eval = np.log(a_eval)
    dx = np.diff(x_eval)
    
    for idx, ref in enumerate(_REF_DATA):
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa = ref

        # Compute model values
        Om_m_eval = _omega_m_of_a(a_eval, Om0, w0, wa)
        gamma_eval = _gamma_from_coeffs(c, a_eval, Om0, w0, wa)

        # Check gamma constraint
        gamma_min = np.min(gamma_eval)
        gamma_max = np.max(gamma_eval)
        if gamma_min < 0.1 or gamma_max > 1.2:
            # Penalize heavily - return large error
            gamma_violation = np.maximum(0.0, 0.1 - gamma_eval) + np.maximum(0.0, gamma_eval - 1.2)
            all_errors.append(10.0 * np.max(gamma_violation))
            continue

        f_model = _f_from_gamma(gamma_eval, Om_m_eval)

        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f
        all_errors.extend(rel_err_f)

        # Compute D_model
        # C(x_i) = cumulative trapezoidal integral of f_model
        f_avg = 0.5 * (f_model[1:] + f_model[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        
        # J(x_i) = C(x_i) - C(x_max)
        J_vals = cum_int - cum_int[-1]
        D_model = np.exp(J_vals)
        D_model = D_model / D_model[-1] # Ensure normalization, should be 1 at end

        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D
        all_errors.extend(rel_err_D)

    if not all_errors:
        return 1e6, []
    
    max_err = max(all_errors)
    return max_err, all_errors

def _objective(c):
    """Compute 100 * max relative error over all validation cosmologies."""
    max_err, _ = _compute_errors_for_coeffs(c)
    return 100.0 * max_err

def _objective_smooth(c, beta=10.0):
    """Smooth surrogate using soft-max (log-sum-exp) for the max operation.
    This makes the objective differentiable and better suited for gradient methods."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    _, all_errors = _compute_errors_for_coeffs(c)
    if not all_errors:
        return 1e6
    
    errors = np.array(all_errors)
    # Soft-max approximation: beta * log(sum(exp(errors/beta)))
    # As beta -> infinity, this approaches max(errors)
    scaled = errors / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = beta * log_sum_exp
    
    return 100.0 * soft_max

# ── Pre-compute references at module level ──────────────────────────────────

_build_references()

# ── Optimize coefficients ───────────────────────────────────────────────────

# Initial guess based on directive: c0=0.55, c3=-0.5
_x0 = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _run_optimization():
    """Run multi-stage optimization with deadline guard.
    
    Strategy:
    1. Global search with multiple diverse starting points using Nelder-Mead
    2. Smooth surrogate optimization with L-BFGS-B for gradient-based refinement
    3. Final high-precision polish with tight tolerances
    """
    best_c = _x0.copy()
    best_err = _objective(best_c)
    
    if not _deadline_ok():
        return best_c

    # Stage 1: Global search with multiple diverse starting points
    rng = np.random.default_rng(42)
    n_starts = 8
    
    for i in range(n_starts):
        if not _deadline_ok():
            break
        
        if i == 0:
            x_start = best_c.copy()
        else:
            # Diverse starting points in [-3, 3]^6
            x_start = rng.uniform(-3, 3, 6)
        
        try:
            result = minimize(
                _objective, x_start, method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Smooth surrogate optimization with L-BFGS-B
    # Use decreasing beta for progressive refinement
    for beta in [50.0, 20.0, 10.0, 5.0]:
        if not _deadline_ok():
            break
        try:
            bounds = [(-5, 5)] * 6
            result = minimize(
                _objective_smooth, best_c, args=(beta,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            # Evaluate with true objective
            true_err = _objective(result.x)
            if true_err < best_err:
                best_err = true_err
                best_c = result.x.copy()
        except Exception:
            pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: High-precision polish with tight tolerances
    # Use very tight tolerances for final refinement
    try:
        result = minimize(
            _objective, best_c, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 4: Additional local perturbations with fine steps
    for _ in range(5):
        if not _deadline_ok():
            break
        x_try = best_c + rng.normal(0, 0.01, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            result = minimize(
                _objective, x_try, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

# Run optimization at import time
_COEFFS = _run_optimization()

def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)
# EVOLVE-BLOCK-END
