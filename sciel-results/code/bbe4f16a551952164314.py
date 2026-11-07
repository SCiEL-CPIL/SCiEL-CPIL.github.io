"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import pi

_start_time = time.time()
_DEADLINE = 205  # seconds, safety margin before 210s kill

def _q_coeffs(c0, c1, c2):
    """
    Compute the 6 coefficients [a5, a4, a3, a2, a1, a0] of q(t) where t = x^2.
    q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    such that g(x) = x^2 * q(x^2).
    """
    # Hermite polynomials (physicist's):
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 10368
    # H_12 = 4096x^12 - 98304x^10 + 1128960x^8 - 5505600x^6 + 11444352x^4 - 10160640x^2 + 103219200
    
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 10368.0
    H12_0 = -103219200.0
    
    # c3 fixed so g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Check: if g tends to negative at infinity, replace g by -g
    # Leading term of g is c3 * 4096 * x^12
    # If c3 < 0, then g -> -inf, so we flip: all coeffs negate
    # But wait - the task says "If g tends to a negative limit at infinity, g is replaced by -g"
    # This means we flip the sign of all coefficients
    
    # Coefficients of g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(x) = (c0) + (-48*c1 - 13440*c2 - 10160640*c3)*x^2 + (16*c1 + 11520*c2 + 11444352*c3)*x^4
    #        + (-3072*c2 - 5505600*c3)*x^6 + (256*c2 + 1128960*c3)*x^8
    #        + (-98304*c3)*x^10 + (4096*c3)*x^12
    
    # q(t) = g(x)/x^2 where t = x^2:
    # q(t) = (-48*c1 - 13440*c2 - 10160640*c3) + (16*c1 + 11520*c2 + 11444352*c3)*t
    #        + (-3072*c2 - 5505600*c3)*t^2 + (256*c2 + 1128960*c3)*t^3
    #        + (-98304*c3)*t^4 + (4096*c3)*t^5
    
    a0 = -48.0*c1 - 13440.0*c2 - 10160640.0*c3
    a1 = 16.0*c1 + 11520.0*c2 + 11444352.0*c3
    a2 = -3072.0*c2 - 5505600.0*c3
    a3 = 256.0*c2 + 1128960.0*c3
    a4 = -98304.0*c3
    a5 = 4096.0*c3
    
    # If leading coefficient is negative, flip all signs (g -> -g)
    if a5 < 0:
        a0, a1, a2, a3, a4, a5 = -a0, -a1, -a2, -a3, -a4, -a5
    
    return [a5, a4, a3, a2, a1, a0]

def _smallest_positive_sign_changing_root(coeffs):
    """
    Find the smallest positive real root of q(t) where q changes sign.
    coeffs = [a5, a4, a3, a2, a1, a0] for q(t) = a5*t^5 + ... + a0
    Returns the root t >= 0, or inf if none exists.
    """
    try:
        roots = np.roots(coeffs)
    except:
        return float('inf')
    
    # Filter for real roots (small imaginary part) that are >= 0
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-9:
            real_val = rt.real
            if real_val >= -1e-9:
                real_roots.append(max(0.0, real_val))
    
    if not real_roots:
        return float('inf')
    
    # Sort and check for sign changes
    real_roots.sort()
    
    def q_eval(t):
        # Horner's method
        result = coeffs[0]
        for c in coeffs[1:]:
            result = result * t + c
        return result
    
    for t in real_roots:
        if t < 1e-12:
            continue
        eps = max(1e-8, t * 1e-8)
        qm = q_eval(max(0.0, t - eps))
        qp = q_eval(t + eps)
        if qm * qp < 0:
            return t
    
    # Fallback: smallest positive root
    for t in real_roots:
        if t > 1e-12:
            return t
    
    return float('inf')

def _objective(c):
    """Objective: minimize r^2/(2*pi) where r^2 = smallest positive sign-changing root of q."""
    c0, c1, c2 = c
    coeffs = _q_coeffs(c0, c1, c2)
    t_root = _smallest_positive_sign_changing_root(coeffs)
    if t_root == float('inf') or t_root < 0:
        return 1e30
    value = t_root / (2.0 * pi)
    return value

def construct():
    """
    Find coefficients that minimize the value r^2/(2*pi).
    Uses scipy optimization: differential evolution for global search,
    then Nelder-Mead for local refinement.
    """
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed.copy()
    best_value = _objective(seed)
    
    # Phase 1: Differential evolution for global search
    bounds = [(-2.0, 2.0), (-0.5, 0.5), (-0.1, 0.1)]
    
    if time.time() - _start_time < _DEADLINE - 60:
        try:
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=200, popsize=20,
                tol=1e-10, mutation=(0.3, 1.0), recombination=0.9,
                polish=False, init='latinhypercube'
            )
            if de_result.fun < best_value:
                best_value = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead local refinement from multiple starting points
    starts = [best_coeffs.copy(), seed.copy()]
    
    # Add some random starts
    rng = np.random.default_rng(123)
    for _ in range(5):
        starts.append(seed + rng.standard_normal(3) * 0.1)
    
    for start in starts:
        if time.time() - _start_time > _DEADLINE - 10:
            break
        try:
            nm_result = minimize(
                _objective, start, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_result.fun < best_value:
                best_value = nm_result.fun
                best_coeffs = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with very small perturbations
    if time.time() - _start_time < _DEADLINE - 5:
        for scale in [1e-14, 1e-13, 1e-12, 1e-11]:
            if time.time() - _start_time > _DEADLINE - 5:
                break
            for i in range(3):
                for sign in [-1, 1]:
                    candidate = best_coeffs.copy()
                    candidate[i] += sign * scale
                    val = _objective(candidate)
                    if val < best_value:
                        best_value = val
                        best_coeffs = candidate.copy()
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
