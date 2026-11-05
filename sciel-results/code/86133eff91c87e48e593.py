"""Improved growth-index approximation with numerical optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad

# Deadline guard
_DEADLINE = None
_START_TIME = None

def _check_deadline():
    """Check if we're approaching the 90s wall-clock limit."""
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        return False
    if _DEADLINE is None:
        _DEADLINE = _START_TIME + 85.0  # 5s safety margin
    return time.time() >= _DEADLINE

def _init_deadline():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # EdS-like normalization: H^2/H0^2 = Om_m a^-3 + Om_de a^(-3(1+w0+wa))
    # where Om_de = 1 - Om0
    # Om(a) = Om_m a^-3 / (Om_m a^-3 + Om_de a^(-3(1+w0+wa)))
    Om_m = Om0
    Om_de = 1.0 - Om0
    if a <= 0:
        return 1.0
    x = a ** (-3.0)
    y = a ** (-3.0 * (1.0 + w0 + wa))
    denom = Om_m * x + Om_de * y
    if denom == 0:
        return 0.0
    return Om_m * x / denom

def _gamma_of_a(a, Om0, w0, wa, coeffs):
    """Compute gamma(a) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a(a, Om0, w0, wa, coeffs):
    """f = Om(a)^gamma"""
    Om = _Om_of_a(a, Om0, w0, wa)
    if Om <= 0:
        return 0.0
    gamma = _gamma_of_a(a, Om0, w0, wa, coeffs)
    return Om ** gamma

def _growth_ode(a, y, Om0, w0, wa):
    """
    Second-order ODE for D(a):
    D'' + (3 + H'/H) D' - (3/2) Om(a) D = 0
    where H'/H = (1/2) d ln(H^2)/d ln a = (1/2)(-3 Om_m a^-3 * H0^2... )
    
    Actually, in terms of a:
    dD/da = v
    dv/da = -3/(2a) v - (1/a)(3/2 * Om(a) - 3/2 * Om(a) ... )
    
    Let me use the standard form:
    D''(a) + (3/2)(1 + Om_de(a) w_de(a))/a * D'(a) - (3/2) Om_m(a)/a * D(a) = 0
    
    Wait, let me be more careful. The growth equation in terms of a is:
    D'' + (3 + f_H)/a D' - (3/2) Om_m / a D = 0
    where f_H = d ln H / d ln a = (1/2) d ln(H^2/H0^2) / d ln a
    
    H^2/H0^2 = Om_m a^{-3} + Om_de a^{-3(1+w0+wa)}
    d ln(H^2/H0^2)/d ln a = -3 Om_m a^{-3} / E^2 - 3(1+w0+wa) Om_de a^{-3(1+w0+wa)} / E^2
    where E^2 = H^2/H0^2
    
    So f_H = (1/2) * [that]
    
    And (3 + f_H)/a is the coefficient of D'.
    """
    E2 = Om0 * a**(-3.0) + (1.0 - Om0) * a**(-3.0*(1.0 + w0 + wa))
    if E2 <= 0:
        return [0.0, 0.0]
    dlnE2_dlna = (-3.0 * Om0 * a**(-3.0) - 3.0 * (1.0 + w0 + wa) * (1.0 - Om0) * a**(-3.0*(1.0 + w0 + wa))) / E2
    f_H = 0.5 * dlnE2_dlna
    D, v = y
    Dpp = -(3.0 + f_H) / a * v + (1.5 * Om0 * a**(-3.0)) / E2 / a * D
    return [v, Dpp]

def _compute_D_ref(a_grid, Om0, w0, wa, a_min=1e-3):
    """Compute normalized D(a)/D(1) by integrating the ODE from a_min to 1, then to a_grid."""
    # Integrate from a_min to 1 with D(a_min)=1, D'(a_min)=f(a_min) (approx)
    # Actually, let's use a standard approach: integrate from a_min to 1
    # Use initial conditions at a_min: D=1, D'=f(a_min)
    a_min = 1e-3
    f_min = _f_of_a(a_min, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])  # rough initial guess
    sol = solve_ivp(_growth_ode, [a_min, 1.0], [1.0, f_min],
                    args=(Om0, w0, wa), rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        return None
    D_at_1 = sol.y[0, -1]
    if D_at_1 == 0:
        return None
    
    # Now integrate from 1.0 to each a in a_grid (or just interpolate)
    # For simplicity, integrate from a_min to max(a_grid) and interpolate
    a_max = max(a_grid)
    if a_max > 1.0:
        sol2 = solve_ivp(_growth_ode, [a_min, a_max], [1.0, f_min],
                         args=(Om0, w0, wa), rtol=1e-10, atol=1e-12, max_step=0.01)
        if not sol2.success:
            return None
        D_vals = sol2.y[0] / D_at_1
        a_vals = sol2.t
    else:
        D_vals = sol.y[0] / D_at_1
        a_vals = sol.t
    
    # Interpolate to get D at a_grid
    D_grid = np.interp(a_grid, a_vals, D_vals)
    return D_grid

def _compute_error(coeffs, models, a_values):
    """Compute 100 * max relative error over all models and a values."""
    max_err = 0.0
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a <= 0:
                continue
            # Compute f
            Om = _Om_of_a(a, Om0, w0, wa)
            if Om <= 0:
                continue
            gamma = _gamma_of_a(a, Om0, w0, wa, coeffs)
            if gamma < 0.1 or gamma > 1.2:
                # Penalty for out-of-range gamma
                max_err = max(max_err, 100.0 * abs(gamma - 0.55))
                continue
            f_approx = Om ** gamma
            
            # Reference f from ODE: f = d ln D / d ln a = a * D' / D
            # We need the reference D and D'
            # For efficiency, compute reference once per model
            # This is expensive; let's use a simpler reference
            # Actually, let's compute the reference f from the ODE solution
            # For now, use the known approximation f_ref = Om(a)^(5/6) as a rough check
            # No, we need the true reference. Let's compute it properly.
            
            # For speed, precompute reference values
            # This function is called many times during optimization, so we need it fast.
            # Let's use a cached reference.
            
    return max_err

# Precompute reference data
_REFERENCE_CACHE = {}

def _get_reference(Om0, w0, wa, a_values):
    """Get precomputed reference f and D values."""
    key = (round(Om0, 6), round(w0, 6), round(wa, 6))
    if key not in _REFERENCE_CACHE:
        a_grid = np.array(a_values)
        # Compute reference D by integrating ODE
        a_min = 1e-3
        # Use a good initial condition: in EdS, D ~ a, so D' ~ 1
        # More accurately, f(a_min) ~ Om(a_min)^(5/6) which is ~1 for small a
        f_min = _Om_of_a(a_min, Om0, w0, wa) ** (5.0/6.0)
        sol = solve_ivp(_growth_ode, [a_min, max(a_grid)], [1.0, f_min],
                        args=(Om0, w0, wa), rtol=1e-10, atol=1e-12, max_step=0.005)
        if not sol.success:
            _REFERENCE_CACHE[key] = None
            return None
        D_vals = sol.y[0]
        Dp_vals = sol.y[1]
        a_vals = sol.t
        D_at_1 = np.interp(1.0, a_vals, D_vals)
        D_norm = D_vals / D_at_1
        # f_ref = a * D' / D
        f_ref = a_vals * Dp_vals / D_vals
        _REFERENCE_CACHE[key] = (a_vals, D_norm, f_ref)
    return _REFERENCE_CACHE[key]

def _compute_error_fast(coeffs, models, a_values):
    """Compute 100 * max relative error using cached references."""
    max_err = 0.0
    for (Om0, w0, wa) in models:
        ref = _get_reference(Om0, w0, wa, a_values)
        if ref is None:
            continue
        a_ref, D_norm_ref, f_ref = ref
        for a in a_values:
            if a <= 0 or a > a_ref[-1]:
                continue
            # Interpolate reference
            f_ref_val = np.interp(a, a_ref, f_ref)
            D_ref_val = np.interp(a, a_ref, D_norm_ref)
            
            # Compute approximation
            Om = _Om_of_a(a, Om0, w0, wa)
            if Om <= 0:
                continue
            gamma = _gamma_of_a(a, Om0, w0, wa, coeffs)
            if gamma < 0.1 or gamma > 1.2:
                max_err = max(max_err, 50.0)
                continue
            f_approx = Om ** gamma
            
            # Relative error in f
            if abs(f_ref_val) > 1e-10:
                err_f = abs(f_approx - f_ref_val) / abs(f_ref_val)
            else:
                err_f = abs(f_approx - f_ref_val)
            max_err = max(max_err, err_f)
            
            # Relative error in D: we need to compute D_approx
            # D_approx/D(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x = ln(a)
            # So D_approx = exp(-int_a^1 f(a') / a' da')
            # We can compute this numerically
            if D_ref_val > 1e-10:
                # Compute D_approx by integrating f_approx from a to 1
                # Use the same grid
                mask = (a_ref >= a) & (a_ref <= 1.0)
                if np.sum(mask) < 2:
                    # Fall back to simple integration
                    x = np.linspace(a, 1.0, 100)
                    f_vals = np.array([_f_of_a(xi, Om0, w0, wa, coeffs) for xi in x])
                    integral = np.trapz(f_vals / x, x)
                    D_approx = np.exp(-integral)
                else:
                    a_sub = a_ref[mask]
                    f_sub = np.array([_f_of_a(xi, Om0, w0, wa, coeffs) for xi in a_sub])
                    integral = np.trapz(f_sub / a_sub, a_sub)
                    D_approx = np.exp(-integral)
                err_D = abs(D_approx - D_ref_val) / abs(D_ref_val)
                max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def construct():
    """Construct the six coefficients for the growth-index approximation."""
    _init_deadline()
    
    # Validation models (typical CPL dark energy models)
    models = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.5),      # CPL with wa=0.5
        (0.3, -1.0, -0.5),     # CPL with wa=-0.5
        (0.3, -0.9, 0.0),      # w0=-0.9
        (0.3, -0.9, 0.3),      # w0=-0.9, wa=0.3
        (0.3, -1.1, 0.0),      # w0=-1.1
        (0.25, -1.0, 0.0),     # Lower Om0
        (0.35, -1.0, 0.0),     # Higher Om0
    ]
    
    a_values = np.linspace(0.05, 1.0, 50)
    
    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_err = _compute_error_fast(best_coeffs, models, a_values)
    
    if _check_deadline():
        return best_coeffs.tolist()
    
    # Optimization
    def objective(x):
        if _check_deadline():
            return best_err
        x = np.clip(x, -10.0, 10.0)
        err = _compute_error_fast(x, models, a_values)
        return err
    
    # Try Nelder-Mead first (robust, no gradient needed)
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'fatol': 1e-6, 'xatol': 1e-6})
        if result.fun < best_err:
            best_err = result.fun
            best_coeffs = np.clip(result.x, -10.0, 10.0)
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeffs.tolist()
    
    # Try L-BFGS-B with numerical gradient
    try:
        result2 = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 6,
                           options={'maxiter': 200, 'ftol': 1e-8})
        if result2.fun < best_err:
            best_err = result2.fun
            best_coeffs = np.clip(result2.x, -10.0, 10.0)
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeffs.tolist()
    
    # Try a few random restarts
    np.random.seed(42)
    for i in range(5):
        if _check_deadline():
            break
        x_rand = x0 + 0.2 * np.random.randn(6)
        x_rand = np.clip(x_rand, -10.0, 10.0)
        try:
            result_r = minimize(objective, x_rand, method='Nelder-Mead',
                               options={'maxiter': 200, 'fatol': 1e-7})
            if result_r.fun < best_err:
                best_err = result_r.fun
                best_coeffs = np.clip(result_r.x, -10.0, 10.0)
        except Exception:
            pass
    
    return np.clip(best_coeffs, -10.0, 10.0).tolist()
# EVOLVE-BLOCK-END
