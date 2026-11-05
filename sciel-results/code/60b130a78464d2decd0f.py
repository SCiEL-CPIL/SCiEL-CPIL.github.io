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
from scipy.integrate import solve_ivp
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
_N_EVAL = len(_EVAL_A)

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

def _objective(c):
    """Compute 100 * max relative error over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0
    a_eval = _EVAL_A
    x_eval = np.log(a_eval)
    dx = np.diff(x_eval)  # length N-1
    
    for idx, ref in enumerate(_REF_DATA):
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa = ref

        # Compute model values
        Om_m_eval = _omega_m_of_a(a_eval, Om0, w0, wa)
        gamma_eval = _gamma_from_coeffs(c, a_eval, Om0, w0, wa)

        # Check gamma constraint
        if np.any(gamma_eval < 0.1) or np.any(gamma_eval > 1.2):
            # Penalize heavily
            gamma_violation = np.maximum(0.0, 0.1 - gamma_eval) + np.maximum(0.0, gamma_eval - 1.2)
            max_err = max(max_err, 10.0 * np.max(gamma_violation))
            continue

        f_model = _f_from_gamma(gamma_eval, Om_m_eval)

        # Relative errors f
        err_f = np.abs(f_model - f_ref) / np.abs(f_ref)
        max_err = max(max_err, np.max(err_f))

        # Compute D_model via integration
        # C(x_i) = \int_{x_min}^{x_i} f(e^t) dt
        # f_avg has length N-1, dx has length N-1
        f_avg = 0.5 * (f_model[1:] + f_model[:-1])  # length N-1
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))  # length N
        # J(x_i) = C(x_i) - C(x_max)
        integral_vals = cum_int - cum_int[-1]  # length N
        D_model = np.exp(integral_vals)
        D_model = D_model / D_model[-1]

        # Relative errors D
        err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_D))

        if not _deadline_ok():
            return max_err

    return 100.0 * max_err

def construct():
    # Deadline guard
    if not _deadline_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Build references
    _build_references()
    
    if not _deadline_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Physically-motivated initial condition:
    # c0=0.55 (matter-era growth index), c3=-0.5 cancels leading w0 term
    # in the growth index expansion, leaving subdominant corrections for
    # the remaining coefficients to model.
    best_coeffs = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _objective(best_coeffs)
    
    # Optimization with multiple restarts if time permits
    def objective(c):
        if not _deadline_ok():
            return best_err
        return _objective(c)
    
    # First pass: Nelder-Mead from physically-motivated start
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except:
        pass
    
    # Second pass: try a few alternative starting points if time allows
    if _deadline_ok():
        alt_starts = [
            [0.55, 0.1, 0.0, -0.5, 0.0, 0.0],
            [0.55, -0.1, 0.0, -0.5, 0.0, 0.0],
            [0.55, 0.0, 0.1, -0.5, 0.0, 0.0],
            [0.55, 0.0, -0.1, -0.5, 0.0, 0.0],
            [0.55, 0.0, 0.0, -0.5, 0.1, 0.0],
            [0.55, 0.0, 0.0, -0.5, -0.1, 0.0],
            [0.55, 0.0, 0.0, -0.5, 0.0, 0.1],
            [0.55, 0.0, 0.0, -0.5, 0.0, -0.1],
        ]
        for alt in alt_starts:
            if not _deadline_ok():
                break
            try:
                result = minimize(objective, np.array(alt), method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
                if result.fun < best_err:
                    best_coeffs = result.x
                    best_err = result.fun
            except:
                pass
    
    # Final refinement from best point
    if _deadline_ok():
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if result.fun < best_err:
                best_coeffs = result.x
                best_err = result.fun
        except:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
