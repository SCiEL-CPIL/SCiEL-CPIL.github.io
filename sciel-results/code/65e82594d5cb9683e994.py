"""Search-based solver for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from sympy import (
    Rational, symbols, Poly, hermite_poly, nsolve,
    sympify, real_roots, sqrt, pi, S, Integer
)
from sympy.polys.polytools import Poly as SymPyPoly
import random
from fractions import Fraction

# Precompute the Hermite polynomial values at 0 and the polynomial structure
_x = symbols('x')
_H0 = hermite_poly(0, _x)
_H4 = hermite_poly(4, _x)
_H8 = hermite_poly(8, _x)
_H12 = hermite_poly(12, _x)

_H0_0 = _H0.subs(_x, 0)  # 1
_H4_0 = _H4.subs(_x, 0)  # 12
_H8_0 = _H8.subs(_x, 0)  # 1008
_H12_0 = _H12.subs(_x, 0)

# Precompute the coefficients of the Hermite polynomials as lists
# H_n(x) = sum_{k=0}^{n} h_{n,k} x^k
# We'll work with the z-polynomial where z = x^2
# g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
# Since all H_n for even n are even polynomials, g(x) is even
# g(x) = x^2 * q(x) where q is a polynomial in x^2 = z
# q(z) is a degree 5 polynomial in z

# Let's extract the coefficients of each H_{2k} in terms of z = x^2
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 53760x^2 + 1008
# H_12(x) = 4096x^12 - 122880x^10 + 1548288x^8 - 10752000x^6 + 37158400x^4 - 53084160x^2 + 103950

# In terms of z = x^2:
# H_0(z) = 1
# H_4(z) = 16z^2 - 48z + 12
# H_8(z) = 256z^4 - 4480z^3 + 26880z^2 - 53760z + 1008
# H_12(z) = 4096z^6 - 122880z^5 + 1548288z^4 - 10752000z^3 + 37158400z^2 - 53084160z + 103950

# But wait, g(x) = x^2 * q(x), so q(x) = g(x)/x^2
# Let's compute the coefficients of g in terms of x, then divide by x^2

# Actually, let's just build the polynomial directly and extract q's coefficients

def _build_q_coeffs(c0_r, c1_r, c2_r):
    """Build the coefficients of q(z) where z = x^2, q is degree 5 in z."""
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    c3_r = -(c0_r * _H0_0 + c1_r * _H4_0 + c2_r * _H8_0) / _H12_0
    
    # Build g as a polynomial in x
    g = c0_r * _H0 + c1_r * _H4 + c2_r * _H8 + c3_r * _H12
    g_poly = SymPyPoly(g, _x)
    
    # Extract coefficients: power -> coeff
    coeff_dict = g_poly.as_dict()
    
    # q(x) = g(x)/x^2, so for each even power p >= 2 in g, q has power p-2
    # Since g is even, all odd powers are zero
    # q is even of degree 10, so q(x) = sum_{j=0}^{5} a_j * x^{2j}
    # Let z = x^2, then q(z) = sum_{j=0}^{5} a_j * z^j
    
    q_coeffs = [Rational(0)] * 6  # a_0 to a_5
    for power in coeff_dict:
        if power[0] >= 2 and power[0] % 2 == 0:
            j = (power[0] - 2) // 2
            q_coeffs[j] = coeff_dict[power]
    
    return q_coeffs, c3_r


def compute_value_exact(c0, c1, c2):
    """
    Given three floats, convert to nearby rationals, compute the value.
    Returns the value r^2/(2*pi) where r is the largest real root of q at which q changes sign.
    If no valid root, returns infinity.
    """
    try:
        # Convert floats to rationals using Fraction for exact float representation
        c0_r = Fraction(c0)
        c1_r = Fraction(c1)
        c2_r = Fraction(c2)
        
        # Build q coefficients
        q_coeffs, c3_r = _build_q_coeffs(c0_r, c1_r, c2_r)
        
        if all(c == 0 for c in q_coeffs):
            return float('inf')
        
        # q(z) = sum_{j=0}^{5} q_coeffs[j] * z^j, where z = x^2
        # We need the largest real root r of q(x) at which q changes sign
        # Since q is even, roots come in pairs +/-r
        # q(x) = q(z) where z = x^2
        # A root r of q(x) corresponds to z = r^2 being a root of q(z)
        # q changes sign at r iff q(z) changes sign at z = r^2 (for z > 0)
        
        # Find positive real roots of q(z)
        z_poly = sum(q_coeffs[j] * _x**j for j in range(6))
        z_poly = SymPyPoly(z_poly, _x)
        
        # Get numerical roots
        roots = z_poly.nroots(method='subresultant', n=15)
        
        # Filter for positive real roots
        positive_real_roots = []
        for root in roots:
            re, im = root.as_real_imag()
            if abs(im) < 1e-10 and re > 0:
                positive_real_roots.append(float(re))
        
        if not positive_real_roots:
            return float('inf')
        
        # For each positive real root z0, check if q(z) changes sign there
        # A root changes sign if it has odd multiplicity
        # We can check by evaluating q(z) just before and after z0
        
        def qz_sign(z_val):
            """Evaluate sign of q(z) at z_val"""
            val = z_poly.evalf(z_val)
            return float(val)
        
        valid_z_roots = []
        for z0 in positive_real_roots:
            eps = 1e-8
            left_val = qz_sign(z0 - eps)
            right_val = qz_sign(z0 + eps)
            
            if left_val * right_val < 0:
                valid_z_roots.append(z0)
        
        if not valid_z_roots:
            return float('inf')
        
        # The largest z0 at which q changes sign
        z_max = max(valid_z_roots)
        
        # r = sqrt(z_max)
        r = np.sqrt(z_max)
        
        # Value = r^2 / (2*pi) = z_max / (2*pi)
        value = z_max / (2 * float(pi))
        
        return value
    
    except Exception as e:
        return float('inf')


def construct():
    """
    Generate candidate coefficients and return the best one found.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s
    
    # Seed
    seed = [-113/100, 1/25, 1/3240]
    best_value = compute_value_exact(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    # If seed value is infinity, we need a different approach
    if best_value == float('inf'):
        # Try some other starting points
        test_points = [
            [0.0, 0.0, 0.0],
            [-0.5, 0.1, 0.01],
            [-1.0, 0.2, 0.02],
            [-0.1, 0.01, 0.001],
        ]
        for tp in test_points:
            v = compute_value_exact(tp[0], tp[1], tp[2])
            if v < best_value:
                best_value = v
                best_coeffs = tp.copy()
    
    # Random search with local refinement
    random.seed(42)
    
    # Define a function to evaluate a candidate
    def evaluate(c0, c1, c2):
        return compute_value_exact(c0, c1, c2)
    
    # We'll do random search around the seed and also explore broader regions
    # Strategy:
    # 1. Local random perturbations around best
    # 2. Broader random sampling
    # 3. Simple gradient-free optimization (e.g., Nelder-Mead like)
    
    # First, do some local random search around the seed
    n_local = 0
    max_local = 500
    
    while time.time() < deadline and n_local < max_local:
        # Perturb around best_coeffs
        scale = 0.1 * (0.5 ** (n_local // 100))  # Decreasing scale
        c0 = best_coeffs[0] + random.gauss(0, scale)
        c1 = best_coeffs[1] + random.gauss(0, scale * 0.1)
        c2 = best_coeffs[2] + random.gauss(0, scale * 0.01)
        
        val = evaluate(c0, c1, c2)
        if val < best_value:
            best_value = val
            best_coeffs = [c0, c1, c2]
            n_local = 0  # Reset counter when we find improvement
        n_local += 1
    
    # Broader random search
    n_broad = 0
    max_broad = 1000
    
    while time.time() < deadline and n_broad < max_broad:
        # Sample in a wider range
        c0 = random.uniform(-2.0, 0.5)
        c1 = random.uniform(-0.5, 0.5)
        c2 = random.uniform(-0.1, 0.1)
        
        val = evaluate(c0, c1, c2)
        if val < best_value:
            best_value = val
            best_coeffs = [c0, c1, c2]
            n_broad = 0  # Reset
        n_broad += 1
    
    # Local refinement using a simple coordinate descent / random perturbation
    n_refine = 0
    max_refine = 2000
    
    scale = 0.01
    while time.time() < deadline and n_refine < max_refine:
        coord = random.randint(0, 2)
        step = random.gauss(0, scale)
        new_coeffs = best_coeffs.copy()
        new_coeffs[coord] += step
        
        val = evaluate(new_coeffs[0], new_coeffs[1], new_coeffs[2])
        if val < best_value:
            best_value = val
            best_coeffs = new_coeffs
            scale *= 0.95  # Slightly decrease scale
            n_refine = 0  # Reset
        else:
            scale *= 0.99
        n_refine += 1
    
    # Final check
    if time.time() > deadline - 5.0:
        # Return best found
        return np.array(best_coeffs)
    
    # One more round of very fine local search
    scale = 0.001
    n_fine = 0
    while time.time() < deadline - 2.0 and n_fine < 500:
        coord = random.randint(0, 2)
        step = random.gauss(0, scale)
        new_coeffs = best_coeffs.copy()
        new_coeffs[coord] += step
        
        val = evaluate(new_coeffs[0], new_coeffs[1], new_coeffs[2])
        if val < best_value:
            best_value = val
            best_coeffs = new_coeffs
            scale *= 0.9
        n_fine += 1
    
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
