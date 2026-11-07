"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

def _evaluate(c):
    """Compute value = r^2 / (2*pi) for given coefficients c = [c0, c1, c2].
    Returns (value, success_flag)."""
    c0, c1, c2 = c
    # Hermite polynomials (physicists'): H_n(x)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
    # H12 = 4096x^12 - 147456x^10 + 1857984x^8 - 9953280x^6 + 21626880x^4 - 18144000x^2 + 10321920
    
    # Values at x=0
    H0_0 = 1
    H4_0 = 12
    H8_0 = 10080
    H12_0 = 10321920
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # q(x) = g(x) / x^2
    # We need q to divide exactly, so g must have x^2 as a factor.
    # g(x) = c0 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080) + c3*(4096x^12 - 147456x^10 + 1857984x^8 - 9953280x^6 + 21626880x^4 - 18144000x^2 + 10321920)
    
    # q(x) = g(x)/x^2 has coefficients for x^0, x^2, x^4, ..., x^10
    # Let's compute q as a polynomial in z = x^2
    # q(x) = a0 + a2*x^2 + a4*x^4 + a6*x^6 + a8*x^8 + a10*x^10
    
    # Coefficient of x^0 in q: from g, the x^2 terms divided by x^2
    # g(x^2 terms): c1*(-48) + c2*(-34560) + c3*(-18144000)
    a0 = c1 * (-48) + c2 * (-34560) + c3 * (-18144000)
    
    # Coefficient of x^2 in q: from g, the x^4 terms divided by x^2
    a2 = c1 * (16) + c2 * (23040) + c3 * (21626880)
    
    # Coefficient of x^4 in q: from g, the x^6 terms divided by x^2
    a4 = c2 * (-4608) + c3 * (-9953280)
    
    # Coefficient of x^6 in q: from g, the x^8 terms divided by x^2
    a6 = c2 * (256) + c3 * (1857984)
    
    # Coefficient of x^8 in q: from g, the x^10 terms divided by x^2
    a8 = c3 * (-147456)
    
    # Coefficient of x^10 in q: from g, the x^12 terms divided by x^2
    a10 = c3 * (4096)
    
    # Check if g tends to negative limit at infinity: leading term is c3*4096*x^12
    # If c3 < 0, g -> -inf, so replace g by -g
    if c3 < 0:
        a0, a2, a4, a6, a8, a10 = -a0, -a2, -a4, -a6, -a8, -a10
    
    # q(z) = a0 + a2*z + a4*z^2 + a6*z^3 + a8*z^4 + a10*z^5 where z = x^2
    # Find largest real root z > 0 where q changes sign
    # Then r = sqrt(z), value = z / (2*pi)
    
    coeffs = [a10, a8, a6, a4, a2, a0]
    roots = np.roots(coeffs)
    
    # Filter for real positive roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10 and root.real > 1e-10:
            real_roots.append(root.real)
    
    if not real_roots:
        return (float('inf'), False)
    
    # Find roots where q changes sign
    # q(z) = a10*z^5 + a8*z^4 + a6*z^3 + a4*z^2 + a2*z + a0
    # Check sign change at each root
    valid_roots = []
    for z in real_roots:
        # Evaluate q at z - eps and z + eps
        eps = 1e-8
        z_minus = max(z - eps, 1e-12)
        z_plus = z + eps
        q_minus = a10*z_minus**5 + a8*z_minus**4 + a6*z_minus**3 + a4*z_minus**2 + a2*z_minus + a0
        q_plus = a10*z_plus**5 + a8*z_plus**4 + a6*z_plus**3 + a4*z_plus**2 + a2*z_plus + a0
        if q_minus * q_plus < 0:
            valid_roots.append(z)
    
    if not valid_roots:
        return (float('inf'), False)
    
    z_max = max(valid_roots)
    value = z_max / (2 * np.pi)
    return (value, True)


def construct():
    """Multi-start Nelder-Mead search for optimal [c0, c1, c2]."""
    deadline = time.time() + 180  # 180s safety margin
    
    # Seed point
    seed = np.array([-113/100, 1/25, 1/3240])
    
    best_c = seed.copy()
    best_val, _ = _evaluate(seed)
    
    # Generate multiple starting points
    np.random.seed(42)
    starts = [seed.copy()]
    
    # Random perturbations around seed
    for i in range(50):
        scale = 1.0 / (1 + i * 0.1)
        perturbation = np.random.normal(0, scale, size=3)
        starts.append(seed + perturbation)
    
    # Additional diverse starting points
    for i in range(20):
        starts.append(np.random.uniform(-2, 2, size=3))
    
    # Sort starts by distance to seed (try closer ones first)
    starts.sort(key=lambda x: np.linalg.norm(x - seed))
    
    for start in starts:
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        try:
            # Nelder-Mead optimization
            result = minimize(
                lambda c: _evaluate(c)[0],
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.success or result.fun < best_val:
                c_opt = result.x
                val_opt, success = _evaluate(c_opt)
                if success and val_opt < best_val:
                    best_val = val_opt
                    best_c = c_opt.copy()
        except Exception:
            continue
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
