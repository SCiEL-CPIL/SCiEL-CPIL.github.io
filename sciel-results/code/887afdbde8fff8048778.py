"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

# Deadline guard: evaluator kills after 90s; target return well before.
_DEADLINE = 55.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return _DEADLINE - _elapsed()

def _H2(a, Om1, w0, wa):
    """H(a)^2/H0^2 for CPL dark energy."""
    if a <= 0:
        return 0.0
    term_matter = Om1 * a**(-3)
    Om_de = 1.0 - Om1
    if abs(Om_de) < 1e-15:
        return term_matter
    # rho_DE(a) = rho_DE0 * a^{-3(1+w0)} * exp(3*wa*(1-a))
    # H2_DE = Om_de * a^{-3(1+w0)} * exp(3*wa*(1-a))
    exponent = -3.0*(1.0 + w0) + 3.0*wa*(1.0 - a)
    if exponent > 500:
        term_de = Om_de * math.exp(500) * a**(-3*(1+w0))
    elif exponent < -500:
        term_de = 0.0
    else:
        term_de = Om_de * math.exp(exponent) * a**(-3*(1+w0))
    return term_matter + term_de

def _compute_reference_f_D(Om1, w0, wa, a_vals):
    """
    Numerically solve the growth ODE to get reference f(a) and D(a)/D(1).
    ODE in x = ln(a):
      d^2D/dx^2 + (3/2 + 1/2 * dlnH/dx) dD/dx - (3/2) Om(a) D = 0
    where Om(a) = H2_matter / H2_total
    """
    # We integrate from x=ln(1)=0 backward to x=ln(a_min)
    # D(1) = 1, D'(1) = f(1) = Om1^gamma... but we need initial conditions.
    # Actually, for the reference, we just need D(a)/D(1) and f(a)=dlnD/dx.
    # Let's use two independent solutions or just integrate forward from small a.
    # Better: integrate from a=1 backward. Set D(1)=1, D'(1)=f(1).
    # But f(1) is what we're trying to approximate. For the REFERENCE, we need
    # the true f and D from the ODE.
    
    # Let's integrate from a small a to a=1, normalizing D(1)=1.
    # Actually, let's just solve the ODE on [ln(a_min), 0] with arbitrary ICs,
    # then normalize.
    
    a_min = 0.05
    x_min = math.log(a_min)
    x_max = 0.0
    
    def ode_system(x, y):
        a = math.exp(x)
        H2_val = _H2(a, Om1, w0, wa)
        if H2_val <= 0:
            H2_val = 1e-30
        Om_a = Om1 * a**(-3) / H2_val
        # dlnH/dx = (1/2) dlnH2/dx
        # H2 = Om1*a^-3 + (1-Om1)*a^{-3(1+w0)}*exp(3*wa*(1-a))
        # dH2/dx = dH2/da * da/dx = dH2/da * a
        # Let's compute numerically for safety
        da_dx = a
        h = 1e-6
        a_plus = a * math.exp(h)
        a_minus = a * math.exp(-h)
        H2_plus = _H2(a_plus, Om1, w0, wa)
        H2_minus = _H2(a_minus, Om1, w0, wa)
        dH2_dx = (H2_plus - H2_minus) / (2*h)
        dlnH_dx = 0.5 * dH2_dx / H2_val
        
        D = y[0]
        Dp = y[1]
        Dpp = -(3.0/2.0 + 0.5*dlnH_dx) * Dp + (3.0/2.0) * Om_a * D
        return [Dp, Dpp]
    
    # Integrate from x_min to x_max
    try:
        sol = solve_ivp(ode_system, [x_min, x_max], [1.0, 0.5], 
                       method='RK45', rtol=1e-10, atol=1e-12,
                       dense_output=True)
        if not sol.success:
            # Fallback: use constant gamma approximation
            gamma_const = 0.55
            f_vals = np.array([Om1 * a**(-3*(1-gamma_const)) / _H2(a, Om1, w0, wa) if _H2(a, Om1, w0, wa) > 0 else 0.5 for a in a_vals])
            # Actually for EdS-like, f ≈ Om(a)^0.55
            f_vals = np.array([max(0.01, (Om1 * a**(-3) / max(1e-10, _H2(a, Om1, w0, wa))))**0.55 for a in a_vals])
            # D: integrate f
            x_vals = np.log(a_vals)
            # D(a)/D(1) = exp(-int_{ln a}^{0} f(e^x) dx)
            D_vals = np.zeros(len(a_vals))
            for i, a in enumerate(a_vals):
                if a >= 1.0:
                    D_vals[i] = 1.0
                else:
                    x_grid = np.linspace(math.log(a), 0.0, 100)
                    f_grid = np.array([max(0.01, (Om1 * math.exp(x)**(-3) / max(1e-10, _H2(math.exp(x), Om1, w0, wa))))**0.55 for x in x_grid])
                    integral = np.trapz(f_grid, x_grid)
                    D_vals[i] = math.exp(-integral)
            return f_vals, D_vals
    except Exception:
        gamma_const = 0.55
        f_vals = np.array([max(0.01, (Om1 * a**(-3) / max(1e-10, _H2(a, Om1, w0, wa))))**0.55 for a in a_vals])
        D_vals = np.zeros(len(a_vals))
        for i, a in enumerate(a_vals):
            if a >= 1.0:
                D_vals[i] = 1.0
            else:
                x_grid = np.linspace(math.log(a), 0.0, 100)
                f_grid = np.array([max(0.01, (Om1 * math.exp(x)**(-3) / max(1e-10, _H2(math.exp(x), Om1, w0, wa))))**0.55 for x in x_grid])
                integral = np.trapz(f_grid, x_grid)
                D_vals[i] = math.exp(-integral)
        return f_vals, D_vals
    
    # Get D and D' at a_vals
    f_vals = np.zeros(len(a_vals))
    D_vals = np.zeros(len(a_vals))
    D1 = sol.y[0, -1]  # D at x=0 (a=1)
    if abs(D1) < 1e-15:
        D1 = 1.0
    
    for i, a in enumerate(a_vals):
        x_a = math.log(a)
        if x_a > x_max:
            x_a = x_max
        if x_a < x_min:
            x_a = x_min
        y_at_a = sol.sol(x_a)
        D_a = y_at_a[0] / D1
        Dp_a = y_at_a[1] / D1
        # f = dlnD/dx = D'/D
        if abs(D_a) > 1e-15:
            f_vals[i] = Dp_a / D_a
        else:
            f_vals[i] = 0.0
        D_vals[i] = D_a
    
    return f_vals, D_vals

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

def _D_normalized(coeffs, a, Om, w0, wa):
    if a <= 0:
        return 0.0
    if a >= 1.0:
        if a == 1.0:
            return 1.0
        # For a > 1, rare
        x_vals = np.linspace(0, math.log(a), 100)
        fs = np.array([_f_value(coeffs, math.exp(x), Om, w0, wa) for x in x_vals])
        integral = np.trapz(fs, x_vals)
        return math.exp(integral)
    
    ln_a = math.log(a)
    x_vals = np.linspace(ln_a, 0.0, 200)
    fs = np.array([_f_value(coeffs, math.exp(x), Om, w0, wa) for x in x_vals])
    integral = np.trapz(fs, x_vals)
    return math.exp(-integral)

def _validate_coeffs(coeffs, a_test_points, Om, w0, wa):
    if not all(np.isfinite(c) for c in coeffs):
        return False
    if not all(-10 <= c <= 10 for c in coeffs):
        return False
    for a in a_test_points:
        gamma = _gamma_from_coeffs(coeffs, a, Om, w0, wa)
        if not (0.1 <= gamma <= 1.2):
            return False
    return True

def _compute_max_error(coeffs, a_vals, Om, w0, wa, f_ref, D_ref):
    max_err = 0.0
    for i, a in enumerate(a_vals):
        Om_a = Om * a**(-3) / max(1e-10, _H2(a, Om, w0, wa))
        f_pred = _f_value(coeffs, a, Om_a, w0, wa)
        f_true = f_ref[i]
        if abs(f_true) > 1e-10:
            err_f = abs(f_pred - f_true) / abs(f_true)
            max_err = max(max_err, err_f)
        
        D_pred = _D_normalized(coeffs, a, Om_a, w0, wa)
        D_true = D_ref[i]
        if abs(D_true) > 1e-10:
            err_D = abs(D_pred - D_true) / abs(D_true)
            max_err = max(max_err, err_D)
    return max_err

def construct():
    global _start_time
    _start_time = time.time()
    
    # Representative CPL cosmologies for optimization
    test_models = [
        (0.30, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.20, -1.2, 0.3),
        (0.40, -0.9, 0.1),
        (0.30, -1.0, 1.0),
        (0.30, -1.0, -1.0),
        (0.35, -1.1, 0.2),
        (0.25, -0.8, -0.3),
    ]
    
    a_vals = np.linspace(0.1, 1.0, 30)
    
    # Precompute references
    refs = []
    for (Om1, w0, wa) in test_models:
        try:
            f_ref, D_ref = _compute_reference_f_D(Om1, w0, wa, a_vals)
            refs.append((Om1, w0, wa, f_ref, D_ref))
        except Exception:
            continue
    
    if len(refs) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Start from seed
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Compute initial error
    def total_error(coeffs):
        max_err = 0.0
        for (Om1, w0, wa, f_ref, D_ref) in refs:
            err = _compute_max_error(coeffs, a_vals, Om1, w0, wa, f_ref, D_ref)
            max_err = max(max_err, err)
        return max_err
    
    best_err = total_error(best_coeffs)
    
    # Simple local search with coordinate descent
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if _remaining() < 5.0:
            break
        improved = True
        while improved and _remaining() > 5.0:
            improved = False
            for i in range(6):
                for direction in [-1, 1]:
                    trial = list(best_coeffs)
                    trial[i] += direction * step
                    if not all(-10 <= c <= 10 for c in trial):
                        continue
                    # Check gamma constraint
                    valid = True
                    for (Om1, w0, wa, f_ref, D_ref) in refs:
                        for a in a_vals:
                            Om_a = Om1 * a**(-3) / max(1e-10, _H2(a, Om1, w0, wa))
                            gamma = _gamma_from_coeffs(trial, a, Om_a, w0, wa)
                            if not (0.1 <= gamma <= 1.2):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    err = total_error(trial)
                    if err < best_err - 1e-12:
                        best_coeffs = trial
                        best_err = err
                        improved = True
                        if _remaining() < 5.0:
                            break
                if _remaining() < 5.0:
                    break
    
    # Final polish with smaller steps
    if _remaining() > 10.0:
        for step in [0.0001, 0.00005, 0.00002]:
            if _remaining() < 5.0:
                break
            for i in range(6):
                for direction in [-1, 1]:
                    trial = list(best_coeffs)
                    trial[i] += direction * step
                    if not all(-10 <= c <= 10 for c in trial):
                        continue
                    valid = True
                    for (Om1, w0, wa, f_ref, D_ref) in refs:
                        for a in a_vals:
                            Om_a = Om1 * a**(-3) / max(1e-10, _H2(a, Om1, w0, wa))
                            gamma = _gamma_from_coeffs(trial, a, Om_a, w0, wa)
                            if not (0.1 <= gamma <= 1.2):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    err = total_error(trial)
                    if err < best_err - 1e-13:
                        best_coeffs = trial
                        best_err = err
    
    # Ensure output is finite and in range
    result = []
    for c in best_coeffs:
        if not np.isfinite(c):
            result.append(0.55)
        else:
            result.append(max(-10.0, min(10.0, c)))
    
    return result
# EVOLVE-BLOCK-END
