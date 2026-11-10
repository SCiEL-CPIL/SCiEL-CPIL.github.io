"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    Omega_DE0 = 1.0 - Om0
    # rho_DE(a)/rho_DE(1) = a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    ratio_DE = (a ** (-3*(1+w0+wa))) * np.exp(3*wa*(a-1))
    Om_a = Om0 * a**(-3) / (Om0 * a**(-3) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    def integrand(x):
        a_x = np.exp(x)
        return _f_of_a(a_x, coeffs, Om0, w0, wa)
    integral_val, _ = quad(integrand, ln_a, 0.0, limit=100)
    return np.exp(-integral_val)

def _solve_growth_reference(a_eval, Om0, w0, wa):
    """Solve the growth ODE numerically to get reference f and D."""
    # Use a fine grid for accuracy
    a_arr = np.linspace(0.05, 1.0, 2000)
    n = len(a_arr)
    
    # Compute Om(a) and q(a) on the grid
    Om_arr = np.array([_Om_of_a(ai, Om0, w0, wa) for ai in a_arr])
    H2_arr = Om_arr * a_arr**(-3)  # H^2/H0^2 for flat universe
    
    # q(a) = 0.5 * d ln H2 / d ln a = 0.5 * a * dH2/da / H2
    q_arr = np.zeros_like(a_arr)
    for i in range(1, n-1):
        da = a_arr[i+1] - a_arr[i-1]
        dH2 = H2_arr[i+1] - H2_arr[i-1]
        q_arr[i] = 0.5 * a_arr[i] * dH2 / (da * H2_arr[i])
    q_arr[0] = q_arr[1]
    q_arr[-1] = q_arr[-2]
    
    # Solve ODE: D' = v, v' = -(3/a - q(a))*v + (3/2)*Om(a)*D
    # Integrate from early times with D~a, v~1 (matter domination)
    D = np.zeros_like(a_arr)
    v = np.zeros_like(a_arr)
    D[0] = a_arr[0]
    v[0] = 1.0
    
    dt = a_arr[1] - a_arr[0]
    for i in range(n - 1):
        ai = a_arr[i]
        Di = D[i]
        vi = v[i]
        qi = q_arr[i]
        Omi = Om_arr[i]
        
        def rhs(D_val, v_val):
            D_prime = v_val
            v_prime = -(3.0/ai - qi) * v_val + (1.5 * Omi) * D_val
            return D_prime, v_prime
        
        k1_D, k1_v = rhs(Di, vi)
        k2_D, k2_v = rhs(Di + 0.5*dt*k1_D, vi + 0.5*dt*k1_v)
        k3_D, k3_v = rhs(Di + 0.5*dt*k2_D, vi + 0.5*dt*k2_v)
        k4_D, k4_v = rhs(Di + dt*k3_D, vi + dt*k3_v)
        
        D[i+1] = Di + (dt/6.0) * (k1_D + 2*k2_D + 2*k3_D + k4_D)
        v[i+1] = vi + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    # Normalize so D(1) = 1
    D_norm = D / D[-1]
    v_norm = v / D[-1]
    
    # Interpolate to find D(a_eval) and f(a_eval) = v/D
    if a_eval <= a_arr[0]:
        return D_norm[0], v_norm[0] / D_norm[0]
    if a_eval >= a_arr[-1]:
        return D_norm[-1], v_norm[-1] / D_norm[-1]
    
    idx = np.searchsorted(a_arr, a_eval)
    t = (a_eval - a_arr[idx-1]) / (a_arr[idx] - a_arr[idx-1])
    D_val = D_norm[idx-1] + t * (D_norm[idx] - D_norm[idx-1])
    v_val = v_norm[idx-1] + t * (v_norm[idx] - v_norm[idx-1])
    f_val = v_val / D_val
    return D_val, f_val

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline for safety
    
    # Test cosmologies (w0, wa, Om0)
    test_models = [
        (-1.0, 0.0, 0.3),      # LCDM
        (-1.0, 0.1, 0.3),      # CPL with wa=0.1
        (-0.9, 0.0, 0.3),      # CPL with w0=-0.9
        (-1.1, 0.0, 0.3),      # CPL with w0=-1.1
        (-1.0, 0.3, 0.3),      # CPL with wa=0.3
        (-0.8, -0.2, 0.3),     # Mixed
        (-1.2, 0.5, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.27),     # LCDM with different Om
        (-0.95, 0.1, 0.28),    # Near-LCDM CPL
        (-1.05, -0.1, 0.32),   # Another near-LCDM
    ]
    
    # a values to check
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Precompute reference solutions for speed
    ref_cache = {}
    for (w0, wa, Om0) in test_models:
        for a in a_values:
            key = (w0, wa, Om0, float(a))
            D_ref, f_ref = _solve_growth_reference(float(a), Om0, w0, wa)
            ref_cache[key] = (D_ref, f_ref)
    
    def objective(coeffs):
        """Max relative error over all test models and a values."""
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            for a in a_values:
                key = (w0, wa, Om0, float(a))
                D_ref, f_ref = ref_cache[key]
                
                # Compute our approximation
                f_approx = _f_of_a(a, coeffs, Om0, w0, wa)
                D_approx = _D_normalized(a, coeffs, Om0, w0, wa)
                
                # Relative errors
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                else:
                    err_f = abs(f_approx - f_ref)
                
                if abs(D_ref) > 1e-10:
                    err_D = abs(D_approx - D_ref) / abs(D_ref)
                else:
                    err_D = abs(D_approx - D_ref)
                
                max_err = max(max_err, err_f, err_D)
        
        return max_err
    
    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_err = objective(x0)
    
    try:
        # Primary optimization with Nelder-Mead
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
        )
        
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
        
        # Check deadline
        if time.time() > deadline - 5.0:
            return np.clip(best_coeffs, -10.0, 10.0).tolist()
        
        # Try multiple starting points for robustness
        starting_points = [
            [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.55, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.5, -0.05, 0.05, -0.05, 0.05, -0.05],
        ]
        
        for sp in starting_points:
            if time.time() > deadline - 5.0:
                break
            sp_arr = np.array(sp)
            res = minimize(
                objective,
                sp_arr,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if res.fun < best_err:
                best_coeffs = res.x
                best_err = res.fun
        
        # Final refinement with tighter tolerance
        if time.time() < deadline - 10.0:
            res_final = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_final.fun < best_err:
                best_coeffs = res_final.x
                best_err = res_final.fun
        
    except Exception:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
