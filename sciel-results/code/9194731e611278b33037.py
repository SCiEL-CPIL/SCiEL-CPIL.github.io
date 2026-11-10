"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, simpson, cumulative_trapezoid
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a_safe))
    return Om0 * a_safe**(-3) / E2

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1 - Om
    u = 1 + w0 + wa * (1 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa, npts=400):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_arr = np.linspace(np.log(ai), 0.0, npts)
            a_arr = np.exp(x_arr)
            f_arr = _f_of_a(a_arr, coeffs, Om0, w0, wa)
            integral = simpson(f_arr, x=x_arr)
            D[i] = np.exp(-integral)
    return D

def _ode_reference_f(a, Om0, w0, wa, npts=800):
    """Solve the GR ODE for f(a) and return f at given a values."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a), 0.01)
    x_min = np.log(a_min)
    x_max = 0.0
    
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5 * Om_at_1 + 0.5 * (1 - Om_at_1) * (1 + w0 + wa)
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5 * (3 * Om - 2) * f + 0.5 * 3 * Om
        return [dfdx]
    
    x_eval = np.sort(a)
    x_eval = x_eval[x_eval > 0]
    if len(x_eval) == 0:
        return np.ones(len(a)) * f0
    
    x_grid = np.linspace(x_min, x_max, npts)
    sol = solve_ivp(ode, [x_min, x_max], [f0], t_eval=x_grid, method='RK45', rtol=1e-10, atol=1e-12)
    
    f_ref = np.zeros(len(a))
    for i, ai in enumerate(a):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                f_ref[i] = sol.y[0, 0]
            elif idx >= len(x_grid):
                f_ref[i] = sol.y[0, -1]
            else:
                x0, x1 = x_grid[idx - 1], x_grid[idx]
                f0v, f1v = sol.y[0, idx - 1], sol.y[0, idx]
                t = (x_val - x0) / (x1 - x0)
                f_ref[i] = f0v + t * (f1v - f0v)
    return f_ref

def _ode_reference_D(a, Om0, w0, wa, npts=800):
    """Solve the GR ODE for D(a) and return normalized D(a)/D(1)."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    
    a_grid = np.exp(x_grid)
    f_grid = _ode_reference_f(a_grid, Om0, w0, wa, npts)
    
    integral_from_min = cumulative_trapezoid(f_grid, x_grid, initial=0.0)
    total = integral_from_min[-1]
    
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                int_val = total - integral_from_min[0]
            elif idx >= len(x_grid):
                int_val = 0.0
            else:
                x0, x1 = x_grid[idx - 1], x_grid[idx]
                i0, i1 = integral_from_min[idx - 1], integral_from_min[idx]
                t = (x_val - x0) / (x1 - x0)
                int_at_x = i0 + t * (i1 - i0)
                int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

def _compute_error(coeffs, Om0, w0, wa, a_vals):
    """Compute max relative error for f and D."""
    f_approx = _f_of_a(a_vals, coeffs, Om0, w0, wa)
    f_ref = _ode_reference_f(a_vals, Om0, w0, wa)
    
    D_approx = _D_normalized(a_vals, coeffs, Om0, w0, wa)
    D_ref = _ode_reference_D(a_vals, Om0, w0, wa)
    
    mask_f = np.abs(f_ref) > 1e-6
    mask_D = np.abs(D_ref) > 1e-6
    
    err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])) if np.any(mask_f) else 0.0
    err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])) if np.any(mask_D) else 0.0
    
    return max(err_f, err_D)

# Validation cosmologies
_VALIDATION_MODELS = [
    (0.30, 0.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
    (0.35, -1.1, 0.2),
]

_A_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

def _objective(coeffs):
    """Objective function: max error across all validation models."""
    if time.time() > _DEADLINE:
        return 1e6
    max_err = 0.0
    for Om0, w0, wa in _VALIDATION_MODELS:
        err = _compute_error(coeffs, Om0, w0, wa, _A_VALUES)
        max_err = max(max_err, err)
    return max_err

def _linder_gamma(Om, w0, wa, a):
    """
    Linder (2005) growth index approximation for CPL dark energy.
    gamma = 1 / (1 + alpha * Om + beta * (1 - Om))
    where alpha and beta are functions of w0, wa.
    
    For CPL: 
      alpha = 0.058 * (1 - w0) + 0.24 * wa
      beta  = 0.019 * (1 - w0) - 0.004 * wa
    (These are approximate; we use a more standard form below)
    
    Actually, the standard Linder form is:
      gamma = 1 / (1 + alpha * Om(a) + beta * (1 - Om(a)))
    
    For CPL parameters, a common parametrization is:
      alpha = 0.058 * (1 - w0) + 0.24 * wa
      beta  = 0.019 * (1 - w0) - 0.004 * wa
    
    But let's use the more accurate form from Linder 2005 eq. (12):
      gamma = 1 / (1 + alpha * Om + beta * (1-Om))
    with alpha, beta determined to match boundary conditions.
    
    A simpler well-known approximation:
      gamma ≈ 0.55 - 0.05 * w0 - 0.01 * wa (roughly constant)
    
    Better: use the analytic solution structure. For CPL, the growth index
    varies with a. We'll use a parametric form that matches known limits.
    """
    # Use a well-known approximation: gamma as function of Om
    # From Linder (2005), for general dark energy:
    # gamma ≈ 1 / (1 + alpha*Om + beta*(1-Om))
    # where for CPL: alpha ≈ 0.058*(1-w0) + 0.24*wa, beta ≈ 0.019*(1-w0) - 0.004*wa
    # But this is an approximation. Let's use a more robust approach.
    
    # Actually, let's just use the known asymptotic + interpolation approach:
    # At a->0: gamma = 6/11
    # At a=1: gamma depends on w0, wa
    
    # For CPL, a good approximation is:
    # gamma(a) = 6/11 + (gamma_1 - 6/11) * (1 - Om(a)) / (1 - Om0)
    # where gamma_1 is the value at a=1
    
    # But we want an analytic formula. Let's use:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # and fit it to the Linder approximation.
    
    # For now, return a reasonable analytic gamma for fitting purposes
    # The Linder approximation for CPL:
    #   gamma ≈ 0.55 + 0.15*(1-w0)*(1-Om) + 0.05*wa*(1-a)
    # This is a rough guide. We'll do a proper fit below.
    
    # Return a placeholder; the real fitting is done in _fit_to_linder
    return None  # Not used directly

def _compute_linder_gamma(Om0, w0, wa, a_grid):
    """
    Compute the Linder (2005) growth index approximation on a grid.
    Uses the standard parametrization for CPL dark energy.
    """
    Om = _Om_of_a(a_grid, Om0, w0, wa)
    q = 1 - Om
    
    # Linder (2005) approximation for CPL:
    # gamma = 1 / (1 + alpha * Om + beta * (1 - Om))
    # with alpha = 0.058 * (1 - w0) + 0.24 * wa
    #      beta  = 0.019 * (1 - w0) - 0.004 * wa
    #
    # However, this simple form may not be accurate for all parameters.
    # A better approach: use the fact that for matter domination gamma=6/11,
    # and interpolate.
    #
    # Let's use a more robust analytic form. For CPL, we can write:
    # gamma(a) = gamma_0 + delta_gamma * (q - q_0) / (1 - q_0)
    # where gamma_0 = 6/11 at q=1 (a=0), and gamma_1 at a=1.
    #
    # Actually, let's just solve the ODE once per model and cache, then fit.
    # But that's what the original code does. Let's use a different strategy:
    # use the known analytic approximation from the literature.
    
    # Standard Linder form:
    alpha = 0.058 * (1 - w0) + 0.24 * wa
    beta = 0.019 * (1 - w0) - 0.004 * wa
    
    gamma = 1.0 / (1.0 + alpha * Om + beta * q)
    
    # Ensure gamma is in reasonable range
    gamma = np.clip(gamma, 0.1, 1.2)
    
    return gamma

def _fit_to_linder():
    """
    Fit the 6-coefficient gamma formula to the Linder approximation
    across all validation models. This is fast (no ODE solving).
    """
    all_A = []
    all_gamma = []
    
    for Om0, w0, wa in _VALIDATION_MODELS:
        a_grid = np.linspace(0.05, 1.0, 200)
        gamma_linder = _compute_linder_gamma(Om0, w0, wa, a_grid)
        
        Om = _Om_of_a(a_grid, Om0, w0, wa)
        q = 1 - Om
        u = 1 + w0 + wa * (1 - a_grid)
        
        A = np.column_stack([
            np.ones_like(a_grid),
            q,
            q**2,
            u,
            wa * a_grid,
            u * q
        ])
        
        all_A.append(A)
        all_gamma.append(gamma_linder)
    
    A_all = np.vstack(all_A)
    gamma_all = np.concatenate(all_gamma)
    
    coeffs, _, _, _ = np.linalg.lstsq(A_all, gamma_all, rcond=None)
    return coeffs

def _optimize():
    """Run optimization with time budget."""
    t0 = time.time()
    deadline = t0 + 80.0
    
    # Step 1: Fast least-squares fit to Linder approximation
    x0 = _fit_to_linder()
    best_x = x0.copy()
    
    # Evaluate the true objective (with ODE reference) at the initial point
    best_val = _objective(x0)
    
    # Step 2: Local refinement using the true objective
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, best_x, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                     'initial_simplex': 0.01 * np.ones(6)})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Step 3: Fine polish
    if time.time() < deadline - 3:
        try:
            result2 = minimize(_objective, best_x, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                                      'initial_simplex': 0.002 * np.ones(6)})
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

_BEST_COEFFS = None

def construct():
    global _BEST_COEFFS
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    else:
        return _optimize()

# Run optimization at import time
try:
    _BEST_COEFFS = _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
