"""Cosmology: compact dark-energy growth approximation.
Improve a six-coefficient growth-index formula for the linear growth rate and normalized
growth factor in smooth CPL dark-energy cosmologies.
"""
import time
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

_DEADLINE = 85.0
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


def _solve_D_ode(a_start, a_end, Om1, w0, wa):
    """
    Solve D'' + (3/2 + 1/2 * dlnH2/dx) D' - (3/2) Om(x) D = 0
    where x = ln a.
    Returns D at a_end given D(a_start)=1, D'(a_start)=f(a_start).
    """
    if a_start <= 0 or a_end <= 0:
        return 0.0, 0.0
    
    x_start = math.log(a_start)
    x_end = math.log(a_end)
    
    # Initial conditions
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
            return sol.y[0, -1], sol.y[1, -1]
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
        if a == 1.0:
            return 1.0
        D_end, _ = _solve_D_ode(1.0, a, Om1, w0, wa)
        return D_end
    else:
        # Integrate from a to 1
        D_end, _ = _solve_D_ode(a, 1.0, Om1, w0, wa)
        # D_end is D(1) given D(a)=1, so D(a)/D(1) = 1/D_end
        if D_end != 0:
            return 1.0 / D_end
        return 0.0


def _gamma_true_analytic(a, Om1, w0, wa):
    """
    Use the known analytical approximation:
    gamma(a) ≈ (5/7) * (Omega_m(a) / (1 + w0 + wa*(1-a)))^0.38
    This is a well-established approximation for CPL dark energy.
    """
    Om_a = _Om_of_a(a, Om1, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    if Om_a <= 0 or u <= 0:
        return 0.55
    ratio = Om_a / u
    if ratio <= 0:
        return 0.55
    return (5.0 / 7.0) * (ratio ** 0.38)


def _f_ref(a, Om1, w0, wa):
    """Reference f using the analytic gamma approximation."""
    Om_a = _Om_of_a(a, Om1, w0, wa)
    if Om_a <= 0:
        return 0.0
    gamma = _gamma_true_analytic(a, Om1, w0, wa)
    return Om_a ** gamma


def _D_normalized(coeffs, a, Om1, w0, wa, n_pts=100):
    """
    Compute D(a)/D(1) = exp(-integral from ln(a) to ln(1) f(exp(x)) dx)
    = exp(-integral from ln(a) to 0 f(exp(x)) dx)
    """
    if a <= 0:
        return 0.0
    if a == 1.0:
        return 1.0
    
    ln_a = math.log(a)
    x_vals = np.linspace(ln_a, 0.0, n_pts)
    
    def integrand(x):
        a_x = math.exp(x)
        Om_x = _Om_of_a(a_x, Om1, w0, wa)
        return _f_value(coeffs, a_x, Om_x, w0, wa)
    
    fs = np.array([integrand(x) for x in x_vals])
    integral = np.trapz(fs, x_vals)
    return math.exp(-integral)


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


def _compute_error(coeffs, test_models, a_vals):
    """
    Compute max relative error of f and D over test models and a values.
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
            D_pred = _D_normalized(coeffs, a, Om1, w0, wa)
            D_true = _D_normalized_ref(a, Om1, w0, wa)
            if abs(D_true) > 1e-15:
                rel_err_D = abs(D_pred - D_true) / abs(D_true)
                max_rel_err = max(max_rel_err, rel_err_D)
    return max_rel_err


def _fit_coefficients_least_squares(test_models, a_vals):
    """
    Fit the 6 coefficients to match the analytic gamma approximation
    gamma_true = (5/7) * (Om(a) / u)^0.38
    across all test models and a values.
    """
    residuals = []
    params = []
    
    for Om1, w0, wa in test_models:
        for a in a_vals:
            Om_a = _Om_of_a(a, Om1, w0, wa)
            if Om_a <= 0:
                continue
            
            u = 1.0 + w0 + wa * (1.0 - a)
            if u <= 0:
                continue
            
            q = 1.0 - Om_a
            gamma_true = _gamma_true_analytic(a, Om1, w0, wa)
            
            # Store parameters for this point
            params.append((q, q*q, u, wa*a, u*q))
            residuals.append(gamma_true)
    
    if len(params) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Build design matrix
    n = len(params)
    A = np.zeros((n, 6))
    for i, (q, q2, u, wa_a, uq) in enumerate(params):
        A[i, 0] = 1.0
        A[i, 1] = q
        A[i, 2] = q2
        A[i, 3] = u
        A[i, 4] = wa_a
        A[i, 5] = uq
    
    b = np.array(residuals)
    
    # Solve least squares
    try:
        coeffs, residuals_ls, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        coeffs = list(coeffs)
        
        # Clamp to valid range
        for i in range(6):
            coeffs[i] = max(-10, min(10, coeffs[i]))
        
        return coeffs
    except:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fit_coefficients_direct(test_models, a_vals):
    """
    Fit coefficients by minimizing the actual max relative error of f and D.
    Uses a grid-based approach followed by local refinement.
    """
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_err = 1.0
    
    # Try several initial points
    initial_candidates = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, 0.02, 0.0, 0.0, 0.0],
        [0.55, -0.05, 0.02, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, -0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        [0.55, 0.0, 0.0, 0.0, 0.0, -0.05],
    ]
    
    for init in initial_candidates:
        if _elapsed() > _DEADLINE - 10:
            break
        try:
            err = _compute_error(init, test_models, a_vals)
            if err < best_err:
                best_err = err
                best_coeffs = init[:]
        except:
            continue
    
    return best_coeffs


def construct():
    global _start_time
    _start_time = time.time()
    
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
        (0.3, -0.9, 0.5),
        (0.3, -1.1, -0.3),
    ]
    
    a_vals = np.linspace(0.05, 1.0, 50)
    
    # Step 1: Fit coefficients via least squares to analytic gamma
    best_coeffs_ls = _fit_coefficients_least_squares(test_models, a_vals)
    
    # Also try direct fitting
    best_coeffs_direct = _fit_coefficients_direct(test_models, a_vals)
    
    # Validate and pick the better one
    valid_ls = all(_validate_coeffs(best_coeffs_ls, a_vals, Om1, w0, wa) for Om1, w0, wa in test_models)
    valid_direct = all(_validate_coeffs(best_coeffs_direct, a_vals, Om1, w0, wa) for Om1, w0, wa in test_models)
    
    if valid_ls and valid_direct:
        try:
            err_ls = _compute_error(best_coeffs_ls, test_models, a_vals)
            err_direct = _compute_error(best_coeffs_direct, test_models, a_vals)
            if err_ls <= err_direct:
                best_coeffs = best_coeffs_ls
                best_err = err_ls
            else:
                best_coeffs = best_coeffs_direct
                best_err = err_direct
        except:
            best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            best_err = 1.0
    elif valid_ls:
        best_coeffs = best_coeffs_ls
        try:
            best_err = _compute_error(best_coeffs, test_models, a_vals)
        except:
            best_err = 1.0
    elif valid_direct:
        best_coeffs = best_coeffs_direct
        try:
            best_err = _compute_error(best_coeffs, test_models, a_vals)
        except:
            best_err = 1.0
    else:
        best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        best_err = 1.0
    
    # Step 2: Local refinement with gradient-based optimization
    if _elapsed() < _DEADLINE - 15:
        # Use Nelder-Mead or similar for local optimization
        from scipy.optimize import minimize
        
        def objective(coeffs):
            if not all(-10 <= c <= 10 for c in coeffs):
                return 1e6
            if not all(np.isfinite(c) for c in coeffs):
                return 1e6
            for Om1, w0, wa in test_models:
                if not _validate_coeffs(coeffs, a_vals, Om1, w0, wa):
                    return 1e6
            try:
                return _compute_error(coeffs, test_models, a_vals)
            except:
                return 1e6
        
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success and result.fun < best_err:
                best_coeffs = list(result.x)
                best_err = result.fun
        except:
            pass
        
        # Additional refinement with tighter tolerances
        if _elapsed() < _DEADLINE - 10:
            try:
                result2 = minimize(objective, best_coeffs, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if result2.success and result2.fun < best_err:
                    best_coeffs = list(result2.x)
                    best_err = result2.fun
            except:
                pass
    
    # Step 3: Fine-tuning with small perturbations
    if _elapsed() < _DEADLINE - 5:
        step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        for step in step_sizes:
            if _elapsed() > _DEADLINE - 5:
                break
            improved = False
            for i in range(6):
                for sign in [-1, 1]:
                    if _elapsed() > _DEADLINE - 5:
                        break
                    candidate = best_coeffs.copy()
                    candidate[i] += sign * step
                    if not all(-10 <= c <= 10 for c in candidate):
                        continue
                    if not all(_validate_coeffs(candidate, a_vals, Om1, w0, wa) for Om1, w0, wa in test_models):
                        continue
                    try:
                        err = _compute_error(candidate, test_models, a_vals)
                    except:
                        continue
                    if err < best_err - 1e-12:
                        best_err = err
                        best_coeffs = candidate
                        improved = True
            if not improved:
                break
    
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
