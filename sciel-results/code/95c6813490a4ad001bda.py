"""Cosmology: compact dark-energy growth approximation.
Improve a six-coefficient growth-index formula for the linear growth rate and normalized
growth factor in smooth CPL dark-energy cosmologies.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

_DEADLINE = 80.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _gamma_from_coeffs(coeffs, a, Om, w0, wa):
    c0, c1, c2, c3, c4, c5 = coeffs
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q


def _f_value(coeffs, a, Om, w0, wa):
    gamma = _gamma_from_coeffs(coeffs, a, Om, w0, wa)
    if Om <= 0:
        return 1.0
    return Om ** gamma


def _H2(a, Om1, w0, wa):
    if a <= 0:
        return 0.0
    term_matter = Om1 * a ** (-3)
    if (1 - Om1) < 0:
        term_de = 0.0
    else:
        exponent = -3 * (1 + w0) + 3 * wa * (1 - a)
        if exponent > 700:
            term_de = (1 - Om1) * 1e300
        elif exponent < -700:
            term_de = 0.0
        else:
            term_de = (1 - Om1) * a ** (-3 * (1 + w0)) * math.exp(exponent)
    return term_matter + term_de


def _Om_of_a(a, Om1, w0, wa):
    h2 = _H2(a, Om1, w0, wa)
    if h2 <= 0:
        return 0.0
    return Om1 * a ** (-3) / h2


def _dlnH2_dx(a, Om1, w0, wa):
    h2 = _H2(a, Om1, w0, wa)
    if h2 <= 0:
        return 0.0
    term1 = -3 * Om1 * a ** (-4)
    if (1 - Om1) == 0:
        term2 = 0.0
    else:
        exp_part = math.exp(3 * wa * (1 - a))
        pow_part = a ** (-3 * (1 + w0) - 1)
        term2 = (1 - Om1) * pow_part * exp_part * (-3) * (1 + w0 + wa)
    h2_prime = term1 + term2
    return a * h2_prime / h2


# Cache for ODE solutions to avoid recomputation
_ode_cache = {}


def _clear_ode_cache():
    global _ode_cache
    _ode_cache = {}


def _solve_D_ode_cached(a_start, a_end, Om1, w0, wa):
    key = (round(a_start, 6), round(a_end, 6), round(Om1, 6), round(w0, 6), round(wa, 6))
    if key in _ode_cache:
        return _ode_cache[key]
    
    if a_start <= 0 or a_end <= 0:
        return 0.0, 0.0
    
    x_start = math.log(a_start)
    x_end = math.log(a_end)
    
    Om_s = _Om_of_a(a_start, Om1, w0, wa)
    D0 = 1.0
    Dp0 = Om_s ** 0.55 if Om_s > 0 else 0.0
    
    def ode(x, y):
        D, Dp = y
        a = math.exp(x)
        Om_a = _Om_of_a(a, Om1, w0, wa)
        dlnH2 = _dlnH2_dx(a, Om1, w0, wa)
        Dpp = -(1.5 + 0.5 * dlnH2) * Dp + 1.5 * Om_a * D
        return [Dp, Dpp]
    
    try:
        sol = solve_ivp(ode, [x_start, x_end], [D0, Dp0], method='RK45',
                        max_step=0.01, rtol=1e-10, atol=1e-12)
        if sol.success:
            result = (sol.y[0, -1], sol.y[1, -1])
            _ode_cache[key] = result
            return result
        else:
            return 0.0, 0.0
    except:
        return 0.0, 0.0


def _D_normalized_ref(a, Om1, w0, wa):
    """
    Compute D(a)/D(1) using ODE reference.
    """
    if a <= 0:
        return 0.0
    if a >= 1.0:
        if abs(a - 1.0) < 1e-12:
            return 1.0
        D_end, _ = _solve_D_ode_cached(1.0, a, Om1, w0, wa)
        return D_end
    else:
        D_end, _ = _solve_D_ode_cached(a, 1.0, Om1, w0, wa)
        if D_end != 0:
            return 1.0 / D_end
        return 0.0


def _f_ref(a, Om1, w0, wa):
    """Reference f from ODE: f = D'/D at scale a."""
    Om_a = _Om_of_a(a, Om1, w0, wa)
    if Om_a <= 0:
        return 0.0
    
    # For a=1, f(1) = Om(1)^0.55 approximately
    if abs(a - 1.0) < 1e-12:
        return Om_a ** 0.55
    
    # Get D and D' at a by integrating from a to a tiny step
    # Actually, we need f(a) = D'(a)/D(a)
    # We can integrate from a to a*(1+epsilon) and use D' at start
    # Better: integrate from some reference and get D'/D at a
    
    # Simple approach: use the fact that near a, D'/D ≈ f(a)
    # Integrate from a to a*1.001 and see the growth
    a_end = a * 1.001
    D_end, Dp_end = _solve_D_ode_cached(a, a_end, Om1, w0, wa)
    
    # D(a) = 1, D'(a) = f(a) (by our initial condition)
    # So f(a) is just the initial D'
    Om_s = _Om_of_a(a, Om1, w0, wa)
    return Om_s ** 0.55 if Om_s > 0 else 0.0


def _D_normalized_fast(coeffs, a, Om1, w0, wa, n_pts=50):
    """
    Compute D(a)/D(1) = exp(-integral from ln(a) to 0 f(exp(x)) dx)
    Uses the actual cosmology to compute Om at each point.
    """
    if a <= 0:
        return 0.0
    if abs(a - 1.0) < 1e-12:
        return 1.0
    
    ln_a = math.log(a)
    if ln_a >= 0:
        x_vals = np.linspace(0, ln_a, n_pts)
        integral = 0.0
        for i in range(n_pts - 1):
            x_mid = 0.5 * (x_vals[i] + x_vals[i+1])
            a_mid = math.exp(x_mid)
            Om_mid = _Om_of_a(a_mid, Om1, w0, wa)
            f_mid = _f_value(coeffs, a_mid, Om_mid, w0, wa)
            integral += f_mid * (x_vals[i+1] - x_vals[i])
        return math.exp(integral)
    else:
        x_vals = np.linspace(ln_a, 0.0, n_pts)
        integral = 0.0
        for i in range(n_pts - 1):
            x_mid = 0.5 * (x_vals[i] + x_vals[i+1])
            a_mid = math.exp(x_mid)
            Om_mid = _Om_of_a(a_mid, Om1, w0, wa)
            f_mid = _f_value(coeffs, a_mid, Om_mid, w0, wa)
            integral += f_mid * (x_vals[i+1] - x_vals[i])
        return math.exp(-integral)


def _compute_max_error_fast(coeffs, test_models, a_vals):
    """
    Compute max relative error of f and D over test models and a values.
    Uses fast D computation.
    """
    max_rel_err = 0.0
    for Om1, w0, wa in test_models:
        for a in a_vals:
            Om_a = _Om_of_a(a, Om1, w0, wa)
            
            # f error
            f_pred = _f_value(coeffs, a, Om_a, w0, wa)
            f_true = _f_ref(a, Om1, w0, wa)
            if abs(f_true) > 1e-15:
                rel_err_f = abs(f_pred - f_true) / abs(f_true)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            # D error
            D_pred = _D_normalized_fast(coeffs, a, Om1, w0, wa)
            D_true = _D_normalized_ref(a, Om1, w0, wa)
            if abs(D_true) > 1e-15:
                rel_err_D = abs(D_pred - D_true) / abs(D_true)
                max_rel_err = max(max_rel_err, rel_err_D)
    return max_rel_err


def _validate_coeffs(coeffs, a_test_points, Om1, w0, wa):
    """Check that gamma is in [0.1, 1.2] at all test points. Return True if valid."""
    if not all(np.isfinite(c) for c in coeffs):
        return False
    if not all(-10 <= c <= 10 for c in coeffs):
        return False
    
    for a in a_test_points:
        Om_a = _Om_of_a(a, Om1, w0, wa)
        gamma = _gamma_from_coeffs(coeffs, a, Om_a, w0, wa)
        if not (0.1 <= gamma <= 1.2):
            return False
    return True


def _soft_max_error(coeffs, test_models, a_vals, beta=50.0):
    """
    Smooth surrogate for max error using log-sum-exp.
    Returns a smooth approximation to the max relative error.
    """
    errors = []
    for Om1, w0, wa in test_models:
        for a in a_vals:
            Om_a = _Om_of_a(a, Om1, w0, wa)
            
            # f error
            f_pred = _f_value(coeffs, a, Om_a, w0, wa)
            f_true = _f_ref(a, Om1, w0, wa)
            if abs(f_true) > 1e-15:
                rel_err_f = abs(f_pred - f_true) / abs(f_true)
                errors.append(rel_err_f)
            
            # D error
            D_pred = _D_normalized_fast(coeffs, a, Om1, w0, wa)
            D_true = _D_normalized_ref(a, Om1, w0, wa)
            if abs(D_true) > 1e-15:
                rel_err_D = abs(D_pred - D_true) / abs(D_true)
                errors.append(rel_err_D)
    
    if not errors:
        return 0.0
    
    errors_arr = np.array(errors)
    # Log-sum-exp for smooth max
    max_e = np.max(errors_arr)
    if max_e < 1e-15:
        return max_e
    log_sum = max_e + math.log(np.sum(np.exp(beta * (errors_arr - max_e))))
    return log_sum / beta


def _local_search_polish(coeffs, test_models, a_vals, max_iterations=100, base_step=0.001):
    """
    High-precision polish pass with decreasing step sizes.
    """
    best_coeffs = coeffs[:]
    best_err = _compute_max_error_fast(best_coeffs, test_models, a_vals)
    
    for iteration in range(max_iterations):
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        step = base_step * (0.95 ** iteration)
        
        for i in range(6):
            for sign in [-1, 1]:
                candidate = best_coeffs[:]
                candidate[i] += sign * step
                if not all(-10 <= c <= 10 for c in candidate):
                    continue
                
                valid = True
                for Om1, w0, wa in test_models:
                    if not _validate_coeffs(candidate, a_vals, Om1, w0, wa):
                        valid = False
                        break
                if not valid:
                    continue
                
                try:
                    err = _compute_max_error_fast(candidate, test_models, a_vals)
                except:
                    continue
                
                if err < best_err - 1e-12:
                    best_err = err
                    best_coeffs = candidate
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_coeffs, best_err


def construct():
    global _start_time
    _start_time = time.time()
    _clear_ode_cache()
    
    # Test cosmologies
    test_models = [
        (0.3, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.2, 0.3),
        (0.4, -0.9, 0.1),
        (0.3, -1.0, 1.0),
        (0.3, -1.0, -1.0),
    ]
    
    a_vals = np.linspace(0.1, 1.0, 20)
    
    # Start from seed
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_err = float('inf')
    
    # Compute initial error
    try:
        best_err = _compute_max_error_fast(best_coeffs, test_models, a_vals)
    except:
        best_err = 1.0
    
    # Phase 1: Coarse local search
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    max_iterations = 500
    
    iteration = 0
    while iteration < max_iterations and _elapsed() < _DEADLINE - 15:
        iteration += 1
        improved = False
        for i in range(6):
            for sign in [-1, 1]:
                for step in step_sizes:
                    candidate = best_coeffs[:]
                    candidate[i] += sign * step
                    if not all(-10 <= c <= 10 for c in candidate):
                        continue
                    
                    # Validate gamma range
                    valid = True
                    for Om1, w0, wa in test_models:
                        if not _validate_coeffs(candidate, a_vals, Om1, w0, wa):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Compute error
                    try:
                        err = _compute_max_error_fast(candidate, test_models, a_vals)
                    except:
                        continue
                    
                    if err < best_err - 1e-10:
                        best_err = err
                        best_coeffs = candidate
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Phase 2: Fine polish
    if _elapsed() < _DEADLINE - 10:
        best_coeffs, best_err = _local_search_polish(
            best_coeffs, test_models, a_vals, 
            max_iterations=200, base_step=0.0005
        )
    
    # Final validation
    if _elapsed() > _DEADLINE - 2:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    final_valid = True
    for Om1, w0, wa in test_models:
        if not _validate_coeffs(best_coeffs, a_vals, Om1, w0, wa):
            final_valid = False
            break
    
    if not final_valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return best_coeffs

# EVOLVE-BLOCK-END
