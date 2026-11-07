"""Optimized program: multi-start Nelder-Mead search for minimal sign-changing root."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 200.0  # safety margin before 210s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k * (2x)^(n-2k) / (k! * (n-2k)!)
# We need coefficients of x^m for m = 0, 2, 4, 6, 8, 10, 12 (even powers only)

def hermite_even_coeffs(n):
    """Return dict mapping power -> coefficient for H_n(x) (even powers only)."""
    coeffs = {}
    for k in range(n // 2 + 1):
        power = n - 2 * k
        if power % 2 != 0:
            continue
        # coefficient = n! * (-1)^k * 2^(n-2k) / (k! * (n-2k)!)
        num = int(factorial(n)) * ((-1) ** k) * (2 ** power)
        den = int(factorial(k)) * int(factorial(power))
        coeffs[power] = num / den
    return coeffs

# Precompute for H0, H4, H8, H12
H0_c = hermite_even_coeffs(0)
H4_c = hermite_even_coeffs(4)
H8_c = hermite_even_coeffs(8)
H12_c = hermite_even_coeffs(12)

# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
# H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-1032192
H0_0 = H0_c.get(0, 0)
H4_0 = H4_c.get(0, 0)
H8_0 = H8_c.get(0, 0)
H12_0 = H12_c.get(0, 0)

def compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) for given coefficients."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) coefficients for even powers
    # g(x) = sum over powers
    g_coeffs = {}
    for power, coeff in H0_c.items():
        g_coeffs[power] = g_coeffs.get(power, 0) + c0 * coeff
    for power, coeff in H4_c.items():
        g_coeffs[power] = g_coeffs.get(power, 0) + c1 * coeff
    for power, coeff in H8_c.items():
        g_coeffs[power] = g_coeffs.get(power, 0) + c2 * coeff
    for power, coeff in H12_c.items():
        g_coeffs[power] = g_coeffs.get(power, 0) + c3 * coeff
    
    # q(x) = g(x) / x^2, so q has powers from 0 to 10 (since g has up to x^12)
    # q(x) = sum_{m even} g_m * x^(m-2)
    q_coeffs = {}
    for power, coeff in g_coeffs.items():
        new_power = power - 2
        if new_power >= 0:
            q_coeffs[new_power] = q_coeffs.get(new_power, 0) + coeff
    
    # q(x) = a0 + a2*x^2 + a4*x^4 + a6*x^6 + a8*x^8 + a10*x^10
    # For sign change: q(x) must change sign at root r
    # Let z = x^2, then q(x) = P(z) where P is polynomial in z
    # But q(x) is NOT purely a function of z^2 in general - it has all even powers
    # Wait: q(x) has powers 0,2,4,6,8,10 which are all even, so q(x) = Q(x^2) where Q is degree 5 in z
    # q(x) = q0 + q2*x^2 + q4*x^4 + q6*x^6 + q8*x^8 + q10*x^10
    # Let z = x^2: q = q0 + q2*z + q4*z^2 + q6*z^3 + q8*z^4 + q10*z^5
    
    q0 = q_coeffs.get(0, 0)
    q2 = q_coeffs.get(2, 0)
    q4 = q_coeffs.get(4, 0)
    q6 = q_coeffs.get(6, 0)
    q8 = q_coeffs.get(8, 0)
    q10 = q_coeffs.get(10, 0)
    
    # Q(z) = q10*z^5 + q8*z^4 + q6*z^3 + q4*z^2 + q2*z + q0
    # Find roots of Q(z), take positive real roots
    # For sign change: Q(z) must change sign at that root (i.e., it's a simple root with odd multiplicity)
    
    z_coeffs = np.array([q10, q8, q6, q4, q2, q0])
    
    # Check if polynomial is essentially zero
    if np.allclose(z_coeffs, 0, atol=1e-30):
        return np.inf
    
    roots = np.roots(z_coeffs)
    
    # Filter: real, positive
    valid_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10 and rt.real > 1e-15:
            z_val = rt.real
            # Check sign change: Q(z) changes sign at z_val
            # Evaluate Q at z_val - eps and z_val + eps
            eps = 1e-8 * max(1.0, abs(z_val))
            Q_minus = q10*(z_val-eps)**5 + q8*(z_val-eps)**4 + q6*(z_val-eps)**3 + q4*(z_val-eps)**2 + q2*(z_val-eps) + q0
            Q_plus = q10*(z_val+eps)**5 + q8*(z_val+eps)**4 + q6*(z_val+eps)**3 + q4*(z_val+eps)**2 + q2*(z_val+eps) + q0
            if Q_minus * Q_plus < 0:
                valid_roots.append(z_val)
    
    if not valid_roots:
        return np.inf
    
    # r = largest real root where q changes sign, value = r^2/(2*pi)
    # Wait: "r be the largest real root of q at which q changes sign"
    # r is the root of q(x), not Q(z). So r = sqrt(z_val)
    # value = r^2 / (2*pi) = z_val / (2*pi)
    # We want to MINIMIZE value, so minimize the LARGEST sign-changing z
    
    # Actually re-reading: "Let r be the largest real root of q at which q changes sign"
    # So r is the LARGEST such root. value = r^2/(2*pi).
    # To minimize value, we want the largest sign-changing root to be as small as possible.
    
    r_max = np.sqrt(max(valid_roots))
    return r_max**2 / (2 * np.pi)

def objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    val = compute_value(c0, c1, c2)
    if np.isinf(val) or np.isnan(val):
        return 1e10
    return val

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = compute_value(seed[0], seed[1], seed[2])
    best_params = seed.copy()
    
    # Multi-start Nelder-Mead
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [seed.copy()]
    
    # Add random perturbations around seed
    for i in range(20):
        scale = 0.5 * (i + 1) / 20
        perturbation = rng.randn(3) * scale
        starts.append(seed + perturbation)
    
    # Add some wider exploration
    for i in range(15):
        scale = 1.0 + 2.0 * i / 15
        perturbation = rng.randn(3) * scale
        starts.append(seed + perturbation)
    
    # Some structured points
    for c0 in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for c1 in [-1, -0.5, 0, 0.5, 1]:
            for c2 in [-0.5, 0, 0.5]:
                if _time_left() < 5:
                    break
                starts.append(np.array([c0, c1, c2]))
    
    nm_options = {
        'xatol': 1e-10,
        'fatol': 1e-12,
        'maxiter': 500,
        'maxfev': 2000,
        'adaptive': True,
    }
    
    for start in starts:
        if _time_left() < 10:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', options=nm_options)
            if res.success or res.fun < best_val:
                val = res.fun
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if _time_left() > 20:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 5000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Second polish
    if _time_left() > 15:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 2000, 'maxfev': 10000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Safety: if time is running out, return best found
    if _time_left() < 5:
        return best_params
    
    # Verify result
    final_val = compute_value(best_params[0], best_params[1], best_params[2])
    if np.isinf(final_val) or np.isnan(final_val) or final_val > best_val * 1.01:
        # Fallback to seed
        return seed
    
    return best_params
# EVOLVE-BLOCK-END
