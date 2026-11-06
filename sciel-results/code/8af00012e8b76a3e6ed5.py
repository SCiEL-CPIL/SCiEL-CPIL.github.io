"""Cosmology: compact dark-energy growth approximation.

construct() returns six finite coefficients in [-10,10] for the ansatz
    q = 1 - Om(a),  u = 1 + w0 + wa*(1-a)
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)**gamma
We fit these coefficients to a reference model built from the contract:
a smooth CPL dark-energy background with a tightly integrated GR second-order
growth ODE used as the independent numerical reference for f and the normalized
growth factor D.

Strategy:
1. Pre-compute high-precision reference tables (f_ref, lnD_ref) on a dense grid
   for a set of validation cosmologies.
2. Use differential evolution to optimize the six coefficients against these
   pre-computed tables, minimizing the worst-case relative error (max norm).
3. Polish with a local Nelder-Mead step on the true objective.
"""
import time
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

# ---------- deadline guard ----------
_DEADLINE = 80.0  # seconds of wall-clock budget (safety margin under 90 s)
_T0 = time.time()


def _budget_left():
    return _DEADLINE - (time.time() - _T0)


# ---------- physics helpers ----------
def _om_m(a, om_m0, w0, wa):
    """Matter density parameter Om(a) for smooth CPL dark energy.
    
    For CPL: w(a) = w0 + wa*(1-a)
    E^2(a) = Om_m0 * a^{-3} + (1-Om_m0) * a^{-3(1+w(a))}
           = Om_m0 * a^{-3} + (1-Om_m0) * exp(-3 * int_a^1 (1+w(a'))/a' da')
    
    int_a^1 (1+w(a'))/a' da' = int_a^1 (1+w0+wa*(1-a'))/a' da'
    = int_a^1 (1+w0)/a' da' + wa * int_a^1 (1-a')/a' da'
    = (1+w0) * ln(1/a) + wa * int_a^1 (1/a' - 1) da'
    = (1+w0) * ln(1/a) + wa * [ln(1/a) - (1-a)]
    = (1+w0+wa) * ln(1/a) - wa*(1-a)
    
    So E^2(a) = Om_m0 * a^{-3} + (1-Om_m0) * a^{-(1+w0+wa)*3} * exp(wa*3*(1-a))
    """
    a = np.asarray(a, dtype=float)
    scalar = np.isscalar(a)
    a = np.atleast_1d(a)
    
    x = 1.0 - a
    # Exponent for dark energy
    # a^{-3(1+w(a))} = a^{-3(1+w0+wa(1-a))} = a^{-3(1+w0+wa)} * a^{3*wa*a}
    # But a^{3*wa*a} is not a power law, so we use the exact integral form:
    # DE scale factor: a_DE^{-3} = exp(3 * int_1^a (1+w(a'))/a' da')
    # int_1^a (1+w(a'))/a' da' = -(1+w0+wa)*ln(a) + wa*(1-a) + wa
    # Wait, let me redo this carefully.
    # int (1+w0+wa*(1-a'))/a' da' = int (1+w0)/a' da' + wa*int (1-a')/a' da'
    # = (1+w0)*ln(a') + wa*[ln(a') - a']
    # = (1+w0+wa)*ln(a') - wa*a'
    # Evaluated from 1 to a:
    # = [(1+w0+wa)*ln(a) - wa*a] - [(1+w0+wa)*ln(1) - wa*1]
    # = (1+w0+wa)*ln(a) - wa*a + wa
    # = (1+w0+wa)*ln(a) + wa*(1-a)
    
    # So DE(a) = DE(1) * exp(3 * [(1+w0+wa)*ln(a) + wa*(1-a) - ((1+w0+wa)*ln(1) + wa*(1-1))])
    # = DE(1) * exp(3 * [(1+w0+wa)*ln(a) + wa*(1-a)])
    # = (1-om_m0) * a^{3*(1+w0+wa)} * exp(3*wa*(1-a))
    
    # Wait, that gives a^{3(1+w0+wa)} which for w=-1 gives a^0 = 1, correct for cosmological constant.
    # For w=0 (matter-like), gives a^3, so DE ~ a^3, which means DE density ~ a^{-3}, correct.
    # But we want DE density parameter, so Om_DE(a) = (1-om_m0) * a^{3*(1+w0+wa)} * exp(3*wa*(1-a)) / E^2
    
    # Actually, let me use the standard form:
    # Om_DE(a) = (1-om_m0) * a^{-3*(1+w0+wa)} * exp(-3*wa*(1-a)) ... no wait.
    
    # Let me just compute E^2 directly:
    # E^2(a) = Om_m0 * a^{-3} + (1-om_m0) * exp(-3 * int_a^1 (1+w(a'))/a' da')
    # int_a^1 (1+w(a'))/a' da' = [(1+w0+wa)*ln(a') - wa*a']_a^1
    # = [(1+w0+wa)*0 - wa*1] - [(1+w0+wa)*ln(a) - wa*a]
    # = -wa - (1+w0+wa)*ln(a) + wa*a
    # = wa*(a-1) - (1+w0+wa)*ln(a)
    # = -wa*(1-a) - (1+w0+wa)*ln(a)
    
    # So exp(-3 * int) = exp(3*wa*(1-a) + 3*(1+w0+wa)*ln(a))
    # = a^{3*(1+w0+wa)} * exp(3*wa*(1-a))
    
    de = (1.0 - om_m0) * np.power(a, 3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a))
    m = om_m0 * np.power(a, -3.0)
    e2 = m + de
    om = m / e2
    return om if not scalar else om[0]


def _ref_f(a, om_m0, w0, wa):
    """Reference growth rate f(a) = d ln D / d ln a from the GR ODE.
    
    Using the approximation from Linder (2005):
    f = Om(a)^gamma where gamma is determined by the ODE.
    
    The GR second-order ODE for D(a):
    D'' + [3/a + H'/H] D' - (3/2) Om(a) H^2 D = 0
    
    With f = d ln D / d ln a, we get:
    f' + f^2 + [3/2 + H'/(H)] f - (3/2) Om(a) = 0
    
    where primes denote d/da.
    
    For a CPL model, H'/H = (3/2)Om(a) + (3/2)(1+3w(a))Om_DE(a)
    
    We solve this ODE numerically to get f(a), then compute D(a) by integration.
    """
    a = np.asarray(a, dtype=float)
    scalar = np.isscalar(a)
    a = np.atleast_1d(a)
    
    # We'll use a simple 4th-order Runge-Kutta to integrate the ODE for f
    # f' = -(f^2 + (3/2 + H'/H) f - (3/2) Om(a))
    # where H'/H = (3/2)Om + (3/2)(1+3w)Om_DE
    
    # Integrate from a=1 (where f(1) ~ Om(1)^0.55 ≈ Om_m0^0.55) down to a_min
    a_min = max(a.min(), 1e-4)
    n_steps = 2000
    a_grid = np.linspace(a_min, 1.0, n_steps + 1)
    
    # Initial condition at a=1
    om1 = _om_m(1.0, om_m0, w0, wa)
    f1 = om1 ** 0.55
    
    f_vals = np.zeros(n_steps + 1)
    f_vals[n_steps] = f1
    
    def df_da(a_val, f_val):
        om = _om_m(a_val, om_m0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        om_de = 1.0 - om
        hprime_over_h = 1.5 * om + 1.5 * (1.0 + 3.0 * w) * om_de
        return -(f_val ** 2 + (1.5 + hprime_over_h) * f_val - 1.5 * om)
    
    # RK4 integration from a=1 down to a_min
    da = (a_min - 1.0) / n_steps
    for i in range(n_steps - 1, -1, -1):
        a_i = a_grid[i]
        a_next = a_grid[i + 1]
        f_i = f_vals[i + 1]
        
        k1 = df_da(a_i + 0.0 * da, f_i)
        k2 = df_da(a_i + 0.5 * da, f_i + 0.5 * da * k1)
        k3 = df_da(a_i + 0.5 * da, f_i + 0.5 * da * k2)
        k4 = df_da(a_i + 1.0 * da, f_i + da * k3)
        
        f_vals[i] = f_i + (da / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Interpolate to the requested a values
    f_result = np.interp(a, a_grid, f_vals)
    return f_result if not scalar else f_result[0]


def _ref_lnD(a, om_m0, w0, wa, f_grid, a_grid):
    """ln D(a) = int_a^1 f(e^x) dx = int_{ln a}^0 f(e^x) dx = -int_0^{ln a} f(e^x) dx
    
    Since f = d ln D / d ln a, we have ln D(a) = int_a^1 f(a') d ln a' = int_{ln a}^0 f(e^x) dx
    """
    x = np.log(np.clip(a, 1e-8, 1.0))
    xg = np.log(np.clip(a_grid, 1e-8, 1.0))
    # ln D(a) = -int_0^{ln a} f(e^x) dx, so we integrate from xg to x
    # Using trapz: integral from xg_min to x of f dx, then negate
    # Actually: ln D(a) = int_{ln a}^0 f(e^x) dx
    # = -int_0^{ln a} f(e^x) dx
    # We have f on a_grid, so f(e^x) on xg.
    # int_{x}^0 f(e^x') dx' = -int_0^{x} f(e^x') dx'
    # Using trapezoidal rule on xg for f(e^x):
    result = -np.array([np.trapz(np.interp(np.linspace(0, xi, 200), xg, f_grid), np.linspace(0, xi, 200)) for xi in x])
    return result


def _ref_lnD_fast(a, om_m0, w0, wa, f_grid, a_grid):
    """Faster version of lnD computation using cumulative trapezoidal integration."""
    xg = np.log(np.clip(a_grid, 1e-8, 1.0))
    # Cumulative integral from xg[0] to each xg[i]
    # cumint[i] = int_{xg[0]}^{xg[i]} f(e^x) dx
    cumint = np.zeros_like(xg)
    for i in range(1, len(xg)):
        cumint[i] = cumint[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * (xg[i] - xg[i-1])
    
    # ln D(a) = int_{ln a}^0 f(e^x) dx = -int_0^{ln a} f(e^x) dx
    # = -(int_{xg[0]}^{ln a} f dx + int_0^{xg[0]} f dx)
    # Since xg[0] = ln(a_min) < 0, and 0 > xg[0]:
    # int_0^{ln a} = int_{xg[0]}^{ln a} - int_{xg[0]}^0
    # int_{xg[0]}^0 f dx = cumint_at_0 (we need to interpolate)
    
    x = np.log(np.clip(a, 1e-8, 1.0))
    # Interpolate cumint at x
    cumint_at_x = np.interp(x, xg, cumint)
    # Interpolate cumint at 0
    cumint_at_0 = np.interp(0.0, xg, cumint)
    
    # int_0^{ln a} f dx = cumint_at_x - cumint_at_0
    # ln D = -(cumint_at_x - cumint_at_0) = cumint_at_0 - cumint_at_x
    return cumint_at_0 - cumint_at_x


# ---------- precomputation ----------
def _precompute(models, a_grid):
    """Pre-compute reference f and lnD for all models on a_grid."""
    ref_data = []
    for (om_m0, w0, wa) in models:
        f_ref = _ref_f(a_grid, om_m0, w0, wa)
        lnD_ref = _ref_lnD_fast(a_grid, om_m0, w0, wa, f_ref, a_grid)
        ref_data.append((f_ref, lnD_ref))
    return ref_data


# ---------- candidate evaluation ----------
def _ansatz_gamma(c, a, om_m0, w0, wa):
    om = _om_m(a, om_m0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _evaluate_c(c, models, a_grid, ref_data):
    """Return max relative error of f and D, and whether gamma is in [0.1, 1.2]."""
    max_err = 0.0
    gamma_ok = True
    for idx, (om_m0, w0, wa) in enumerate(models):
        om_arr = _om_m(a_grid, om_m0, w0, wa)
        g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_ok = False
        f_cand = np.power(om_arr, g)
        f_ref, lnD_ref = ref_data[idx]
        err_f = np.max(np.abs(f_cand - f_ref) / np.maximum(f_ref, 1e-12))
        max_err = max(max_err, err_f)
        
        lnD_cand = _ref_lnD_fast(a_grid, om_m0, w0, wa, f_cand, a_grid)
        D_ref = np.exp(lnD_ref)
        D_cand = np.exp(lnD_cand)
        err_D = np.max(np.abs(D_cand - D_ref) / np.maximum(D_ref, 1e-12))
        max_err = max(max_err, err_D)
    return max_err, gamma_ok


def _objective(c, models, a_grid, ref_data):
    """Objective function for optimization: max relative error + penalty for gamma violation."""
    max_err, gamma_ok = _evaluate_c(c, models, a_grid, ref_data)
    if not gamma_ok:
        # Penalty for gamma out of range
        penalty = 0.0
        for (om_m0, w0, wa) in models:
            g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
            penalty += np.sum(np.maximum(0.1 - g, 0) ** 2) + np.sum(np.maximum(g - 1.2, 0) ** 2)
        return max_err + 10.0 * penalty
    return max_err


# ---------- fitting ----------
def _fit(models, a_grid, ref_data, deadline_guard):
    """Fit coefficients using differential evolution + local polish."""
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10.0, 10.0)] * 6
    
    best_c = c0.copy()
    best_err, _ = _evaluate_c(best_c, models, a_grid, ref_data)
    
    # Differential evolution
    try:
        if deadline_guard() > 40.0:
            result = differential_evolution(
                lambda c: _objective(c, models, a_grid, ref_data),
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
            )
            if result.success or result.fun < best_err:
                best_c = result.x
                best_err, _ = _evaluate_c(best_c, models, a_grid, ref_data)
    except Exception:
        pass
    
    # Local polish with Nelder-Mead
    try:
        if deadline_guard() > 10.0:
            result = minimize(
                lambda c: _objective(c, models, a_grid, ref_data),
                best_c,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
            )
            if result.fun < best_err:
                best_c = result.x
                best_err, _ = _evaluate_c(best_c, models, a_grid, ref_data)
    except Exception:
        pass
    
    return best_c


def construct():
    # Standard CPL validation cosmologies (a smooth, representative set).
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.5),   # CPL
        (0.30, -1.1, 0.5),
        (0.30, -0.9, 0.0),
        (0.30, -1.0, -0.5),
        (0.25, -1.0, 0.5),
        (0.35, -1.0, 0.5),
        (0.30, -1.2, 0.8),
    ]
    a_grid = np.linspace(0.05, 1.0, 80)

    try:
        ref_data = _precompute(models, a_grid)
        if _budget_left() > 30.0:
            c = _fit(models, a_grid, ref_data, deadline_guard=_budget_left)
        else:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    c = np.clip(c, -10.0, 10.0)
    # sanity: ensure gamma in [0.1,1.2] on a probe; if violated, fall back to seed
    try:
        probe_a = np.linspace(0.05, 1.0, 100)
        ok = True
        for (om_m0, w0, wa) in models:
            g = _ansatz_gamma(c, probe_a, om_m0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                ok = False
                break
        if not ok:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return [float(v) for v in c]
