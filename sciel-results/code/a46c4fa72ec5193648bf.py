"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # safety margin within 210s limit

def _time_remaining():
    return _DEADLINE - time.time()

# Precompute Hermite polynomial coefficients (physicists' Hermite) in ascending order
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 560
# H_12(x) = 4096x^12 - 114688x^10 + 1336320x^8 - 6126720x^6 + 11010048x^4 - 6935040x^2 + 1034880

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 560.0
H12_0 = 1034880.0

# Coefficients in ascending order (index = degree)
H4_coeffs = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
H8_coeffs = np.array([560.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
H12_coeffs = np.array([1034880.0, 0.0, -6935040.0, 0.0, 11010048.0, 0.0, -6126720.0, 0.0, 1336320.0, 0.0, -114688.0, 0.0, 4096.0])

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    try:
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build g coefficients (ascending order), degree up to 12
        g_coeffs = np.zeros(13)
        g_coeffs[0] += c0 * H0_0
        for i in range(5):
            g_coeffs[i] += c1 * H4_coeffs[i]
        for i in range(9):
            g_coeffs[i] += c2 * H8_coeffs[i]
        for i in range(13):
            g_coeffs[i] += c3 * H12_coeffs[i]
        
        # g(0) should be 0, so g_coeffs[0] should be ~0
        # q(x) = g(x) / x^2, so q_coeffs = g_coeffs[2:]
        q_coeffs = g_coeffs[2:]  # degree 10 polynomial
        
        # Find real roots of q
        roots = np.roots(q_coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        # Check if g tends to negative limit at infinity
        # Leading coefficient of g is c3 * 4096
        leading_coeff = g_coeffs[12]
        if leading_coeff < 0:
            q_coeffs = -q_coeffs
            # Re-find roots (sign flip doesn't change roots, but we need to re-evaluate)
            # Actually roots are the same, just signs of q flip
        
        # Find sign-changing positive roots
        sign_changing_roots = []
        for r in real_roots:
            if r <= 0:
                continue
            eps = 1e-8 * max(1.0, abs(r))
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                sign_changing_roots.append(r)
        
        if not sign_changing_roots:
            return 0.0
        
        r_max = max(sign_changing_roots)
        return r_max ** 2 / (2.0 * math.pi)
    except Exception:
        return float('inf')

def _objective(coeffs):
    """Objective function for optimization."""
    if _time_remaining() < 5:
        return 1e10
    c0, c1, c2 = coeffs
    return _compute_value(c0, c1, c2)

def construct():
    best_value = _compute_value(-113/100, 1/25, 1/3240)
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    # Phase 1: Differential evolution for global search
    if _time_remaining() > 120:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=20,
                tol=1e-6, seed=42,
                polish=False,
                callback=lambda x, convergence: None
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starting points
    start_points = [
        best_coeffs,
        np.array([-1.13, 0.04, 0.00030864]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.13, 0.041, 0.00031]),
        np.array([-1.13, 0.039, 0.00030]),
    ]
    
    for x0 in start_points:
        if _time_remaining() < 15:
            break
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 3: Final tight refinement
    if _time_remaining() > 10:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Return the best coefficients found
    if _time_remaining() < 0:
        return np.array([-113/100, 1/25, 1/3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
