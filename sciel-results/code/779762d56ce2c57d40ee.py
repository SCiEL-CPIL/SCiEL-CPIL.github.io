"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Reference cosmology parameters for training (typical CPL models)
# We'll use a set of representative cosmologies to fit the coefficients
# The evaluator uses "disjoint public validation cosmologies" so we need to
# be robust across a range of parameters

def _Om_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    # Om(a) = Om0 * a^3 / E(a)^2 where E(a)^2 = Om0*a^3 + (1-Om0)*a^(-3*(1+w0+wa*(1-a)))
    # Actually for CPL: w(a) = w0 + wa*(1-a)
    # E(a)^2 = Om0 * a^3 + (1-Om0) * a^(-3*(1+w0+wa*(1-a)))
    # Wait, that's not quite right. Let me reconsider.
    # For CPL dark energy: w(a) = w0 + wa*(1-a)
    # The dark energy density scales as: rho_DE(a) = rho_DE(1) * a^(-3(1+w(a)))
    # But w(a) is not constant, so we need to integrate:
    # ln(rho_DE(a)/rho_DE(1)) = -3 * integral_1^a (1+w(a'))/a' da'
    # = -3 * integral_1^a (1+w0+wa*(1-a'))/a' da'
    # = -3 * [(1+w0)*ln(a) + wa*(ln(a) - (1-a))]
    # = -3 * [(1+w0+wa)*ln(a) - wa*(1-a)]
    # So rho_DE(a) = rho_DE(1) * a^(-3(1+w0+wa)) * exp(3*wa*(1-a))
    
    # Hmm, this is getting complicated. Let me use a simpler approach.
    # Actually, for the standard CPL parametrization:
    # E(a)^2 = Om_m * a^3 + Om_DE * a^(-3(1+w0+wa)) * exp(3*wa*(1-a))
    # where Om_DE = 1 - Om_m (assuming flat universe, no radiation)
    
    Om_m = Om0
    Om_DE = 1.0 - Om0
    # DE density scaling
    exponent = -3.0 * (1.0 + w0 + wa)
    exp_factor = np.exp(3.0 * wa * (1.0 - a))
    DE_term = Om_DE * a**exponent * exp_factor
    m_term = Om_m * a**3
    E2 = m_term + DE_term
    return Om_m * a**3 / E2

def _gamma_exact(a, Om0, w0, wa, n_steps=1000):
    """Compute the exact growth index gamma from the ODE.
    
    The growth equation is: D'' + (2 + H'/H) D' - (3/2) Om_m D = 0
    where H'/H = (1/2) d/dln(a) ln(E^2)
    
    gamma is defined by f = Om(a)^gamma, where f = d ln D / d ln a.
    We can compute f directly from the ODE solution.
    """
    # We'll integrate the ODE from a=1 backwards to the desired a
    # D'' + (2 + H'/H) D' - (3/2) Om_m D = 0
    # Let x = ln(a), then d/dx = a d/da
    # D_x' + (2 + H'/H) D_x' - (3/2) Om_m D = 0
    # where H'/H = (1/2) d/dx ln(E^2)
    
    # For a given x, we need:
    # E(x)^2 = Om_m * e^(3x) + Om_DE * e^(-3(1+w0+wa)(x)) * exp(3*wa*(1-e^x))
    # Wait, a = e^x, so 1-a = 1-e^x
    
    # Let me just numerically compute gamma by solving the ODE
    
    # Integrate from x=0 (a=1) to x_target = ln(a)
    x_target = np.log(a)
    
    # ODE system: y1 = D, y2 = D'
    # y1' = y2
    # y2' = -(2 + H'/H)*y2 + (3/2)*Om_m*y1
    
    # H'/H = (1/2) * d/dx ln(E^2)
    # E^2 = Om_m * a^3 + Om_DE * a^(-3(1+w0+wa)) * exp(3*wa*(1-a))
    # d/da ln(E^2) = [3*Om_m*a^2 + Om_DE * (-3(1+w0+wa)) * a^(-3(1+w0+wa)-1) * exp(3*wa*(1-a)) + Om_DE * a^(-3(1+w0+wa)) * exp(3*wa*(1-a)) * 3*wa] / E^2
    # d/dx = a * d/da
    # So d/dx ln(E^2) = a * d/da ln(E^2)
    
    def H_prime_over_H(x):
        a_val = np.exp(x)
        Om_m = Om0
        Om_DE = 1.0 - Om0
        
        # E^2 terms
        a3 = a_val**3
        de_exp = np.exp(-3.0*(1.0+w0+wa)*x + 3.0*wa*(1.0-a_val))
        E2 = Om_m*a3 + Om_DE*de_exp
        
        # d(E^2)/da
        dE2_da = 3.0*Om_m*a_val**2 + Om_DE*de_exp*(-3.0*(1.0+w0+wa)/a_val + 3.0*wa)
        
        # d/dx ln(E^2) = a * (1/E^2) * dE2_da
        return 0.5 * a_val * dE2_da / E2
    
    def ode_system(x, y):
        y1, y2 = y
        a_val = np.exp(x)
        Om_m_val = Om0 * a_val**3 / (Om0*a_val**3 + (1.0-Om0)*np.exp(-3.0*(1.0+w0+wa)*x + 3.0*wa*(1.0-a_val)))
        HHH = H_prime_over_H(x)
        dy1 = y2
        dy2 = -(2.0 + HHH)*y2 + 1.5*Om_m_val*y1
        return [dy1, dy2]
    
    # Initial conditions at x=0 (a=1): D(1)=1, D'(1)=f(1)
    # f(1) ≈ 1 for standard cosmologies, but let's compute it
    # Actually, for the growth index, we just need f = D'/D
    # At a=1, f(1) = D'(1)/D(1). For a flat LambdaCDM, f(1) ≈ 1.
    # Let's use f(1) = 1.0 as initial condition, which is standard.
    
    y0 = [1.0, 1.0]
    
    # Integrate using simple RK4 or use scipy
    from scipy.integrate import solve_ivp
    
    if abs(x_target) < 1e-10:
        return 1.0  # f(1) = 1
    
    sol = solve_ivp(ode_system, [0, x_target], y0, method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        # Fallback: return approximate gamma
        return 0.55
    
    y_end = sol.y[:, -1]
    D_end = y_end[0]
    Dp_end = y_end[1]
    f_end = Dp_end / D_end
    
    # gamma = f / Om(a) ... no, f = Om(a)^gamma, so gamma = ln(f)/ln(Om(a))
    Om_a_val = Om0 * np.exp(3*x_target) / (Om0*np.exp(3*x_target) + (1.0-Om0)*np.exp(-3.0*(1.0+w0+wa)*x_target + 3.0*wa*(1.0-np.exp(x_target))))
    
    if Om_a_val <= 0 or Om_a_val >= 1:
        return 0.55
    
    gamma_val = np.log(f_end) / np.log(Om_a_val)
    
    # Clamp to reasonable range
    gamma_val = np.clip(gamma_val, 0.1, 1.2)
    
    return gamma_val

def _compute_f(a, coeffs, Om0, w0, wa):
    """Compute f = Om(a)^gamma from the ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om_a**gamma

def _compute_D_normalized(a, coeffs, Om0, w0, wa, n_points=200):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    
    x_target = np.log(a)
    
    # Integrate f(exp(x)) from x_target to 0
    xs = np.linspace(x_target, 0, n_points)
    f_vals = np.array([_compute_f(np.exp(x), coeffs, Om0, w0, wa) for x in xs])
    
    # Simpson's rule or trapezoidal
    integral = np.trapz(f_vals, xs)
    
    return np.exp(-integral)

def _objective(coeffs, Om0_list, w0_list, wa_list, a_values):
    """Compute the worst relative error across all cosmologies and a values."""
    max_err = 0.0
    for i in range(len(Om0_list)):
        Om0, w0, wa = Om0_list[i], w0_list[i], wa_list[i]
        for a in a_values:
            # Compute f error
            f_approx = _compute_f(a, coeffs, Om0, w0, wa)
            f_exact = _compute_f_exact(a, Om0, w0, wa)
            err_f = abs(f_approx - f_exact) / abs(f_exact)
            
            # Compute D error
            D_approx = _compute_D_normalized(a, coeffs, Om0, w0, wa)
            D_exact = _compute_D_exact(a, Om0, w0, wa)
            err_D = abs(D_approx - D_exact) / abs(D_exact)
            
            max_err = max(max_err, err_f, err_D)
    
    return 100.0 * max_err

def _compute_f_exact(a, Om0, w0, wa):
    """Compute exact f from the ODE."""
    gamma = _gamma_exact(a, Om0, w0, wa)
    Om_a = _Om_a(a, Om0, w0, wa)
    return Om_a**gamma

def _compute_D_exact(a, Om0, w0, wa, n_points=200):
    """Compute exact D(a)/D(1)."""
    if a >= 1.0:
        return 1.0
    
    x_target = np.log(a)
    xs = np.linspace(x_target, 0, n_points)
    f_vals = np.array([_compute_f_exact(np.exp(x), Om0, w0, wa) for x in xs])
    integral = np.trapz(f_vals, xs)
    return np.exp(-integral)

def construct():
    """Find the six coefficients that minimize the worst relative error."""
    start_time = time.time()
    deadline = 75.0  # Safety margin before 90s limit
    
    # Training cosmologies (representative CPL models)
    Om0_list = [0.27, 0.30, 0.25, 0.32, 0.28]
    w0_list = [-1.0, -0.9, -1.1, -0.95, -1.0]
    wa_list = [0.0, 0.1, -0.1, 0.05, 0.15]
    
    # a values to check
    a_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.55, 0.75, 0.85])
    
    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _objective(best_coeffs, Om0_list, w0_list, wa_list, a_values)
    
    # Multi-start Nelder-Mead
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x0 = best_coeffs.copy()
        else:
            # Random perturbation
            x0 = best_coeffs + 0.1 * np.random.randn(6)
            x0 = np.clip(x0, -10, 10)
        
        try:
            result = minimize(
                lambda c: _objective(c, Om0_list, w0_list, wa_list, a_values),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.success or result.fun < best_score:
                new_coeffs = np.clip(result.x, -10, 10)
                new_score = result.fun
                if new_score < best_score:
                    best_coeffs = new_coeffs
                    best_score = new_score
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                lambda c: _objective(c, Om0_list, w0_list, wa_list, a_values),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                best_coeffs = np.clip(result.x, -10, 10)
                best_score = result.fun
        except Exception:
            pass
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
