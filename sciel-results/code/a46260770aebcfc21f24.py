"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq
import time


def _compute_r(c0, c1, c2):
    """Compute r: largest sign-changing real root of q = g/x^2.
    
    g = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0.
    q = g/x^2 must divide exactly.
    """
    # Hermite polynomials (physicists' convention): H_n(x)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5242880x^6 + 12451840x^4 - 12058624x^2 + 3991680
    
    # g(x) = c0*1 + c1*(16x^4 - 48x^2 + 12) + c2*(256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680) + c3*(4096x^12 - 98304x^10 + 1032192x^8 - 5242880x^6 + 12451840x^4 - 12058624x^2 + 3991680)
    
    # g(0) = c0 + 12*c1 + 1680*c2 + 3991680*c3 = 0
    # c3 = -(c0 + 12*c1 + 1680*c2) / 3991680
    
    c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 3991680.0
    
    # g(x) / x^2 = q(x), which is a polynomial of degree 10
    # g(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    
    a12 = 4096.0 * c3
    a10 = -98304.0 * c3
    a8 = 1032192.0 * c3 + 256.0 * c2
    a6 = -5242880.0 * c3 - 3072.0 * c2
    a4 = 12451840.0 * c3 + 11520.0 * c2 + 16.0 * c1
    a2 = -12058624.0 * c3 - 13440.0 * c2 - 48.0 * c1
    a0 = 3991680.0 * c3 + 1680.0 * c2 + 12.0 * c1 + c0  # should be 0 by construction
    
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    # Let u = x^2, then q = a12*u^5 + a10*u^3 + a8*u^2 + a6*u + a4*u + a2... 
    # Wait, let me re-express: q(x) in terms of u = x^2:
    # q = a12*u^5 + a10*u^3 + a8*u^2 + a6*u + a4*u + a2? No.
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    # In u = x^2: q = a12*u^5 + a10*u^3 + a8*u^2 + a6*u + a4*u + a2
    # Hmm, that's not right either. Let me be careful:
    # x^10 = u^5, x^8 = u^4, x^6 = u^3, x^4 = u^2, x^2 = u
    # So q(u) = a12*u^5 + a10*u^4 + a8*u^3 + a6*u^2 + a4*u + a2
    
    # We want the largest real root r of q(x) where q changes sign.
    # Since q is even (only even powers of x), roots come in ± pairs.
    # We want the largest positive root.
    
    # Define q(x) as a function
    def q_func(x):
        x2 = x*x
        x4 = x2*x2
        x6 = x4*x2
        x8 = x4*x4
        x10 = x8*x2
        return a12*x10 + a10*x8 + a8*x6 + a6*x4 + a4*x2 + a2
    
    # Find all real roots of q(x) = 0
    # Use numpy roots on the polynomial coefficients
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    # Coefficients in descending order:
    coeffs = np.array([a12, 0, a10, 0, a8, 0, a6, 0, a4, 0, a2])
    
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None
    
    # Filter for real roots (small imaginary part)
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            real_roots.append(root.real)
    
    if not real_roots:
        return None
    
    # Find the largest positive root where q changes sign
    # For a polynomial, a root is sign-changing if the multiplicity is odd
    # We'll check by evaluating q on both sides of each candidate root
    
    positive_roots = [r for r in real_roots if r > 1e-10]
    if not positive_roots:
        return None
    
    positive_roots.sort(reverse=True)
    
    for r in positive_roots:
        # Check if q changes sign at r
        eps = 1e-8 * max(1.0, abs(r))
        q_left = q_func(r - eps)
        q_right = q_func(r + eps)
        if q_left * q_right < 0:
            return r
    
    return None


def _value_from_r(r):
    """Compute value = r^2 / (2*pi)"""
    return r*r / (2.0 * np.pi)


def construct():
    """Find coefficients [c0, c1, c2] that minimize value = r^2/(2*pi)."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Seed values
    c0_seed = -113.0 / 100.0
    c1_seed = 1.0 / 25.0
    c2_seed = 1.0 / 3240.0
    
    # Compute seed value
    r_seed = _compute_r(c0_seed, c1_seed, c2_seed)
    if r_seed is None:
        r_seed = 1.53  # Approximate
    best_r = r_seed
    best_coeffs = np.array([c0_seed, c1_seed, c2_seed])
    
    # Grid search around the seed
    # Range: explore c0 in [-2, 0], c1 in [0, 1], c2 in [0, 0.1]
    # Coarse grid first, then refine
    
    # Phase 1: Coarse grid
    c0_vals = np.linspace(-2.0, 0.0, 25)
    c1_vals = np.linspace(0.0, 1.0, 25)
    c2_vals = np.linspace(0.0, 0.1, 15)
    
    for c0 in c0_vals:
        for c1 in c1_vals:
            for c2 in c2_vals:
                if time.time() > deadline - 30.0:
                    break
                r = _compute_r(c0, c1, c2)
                if r is not None and r < best_r:
                    best_r = r
                    best_coeffs = np.array([c0, c1, c2])
    
    # Phase 2: Fine grid around best found
    if time.time() < deadline - 60.0:
        c0_center, c1_center, c2_center = best_coeffs
        c0_fine = np.linspace(c0_center - 0.5, c0_center + 0.5, 30)
        c1_fine = np.linspace(max(0, c1_center - 0.3), c1_center + 0.3, 30)
        c2_fine = np.linspace(max(0, c2_center - 0.05), c2_center + 0.05, 20)
        
        for c0 in c0_fine:
            for c1 in c1_fine:
                for c2 in c2_fine:
                    if time.time() > deadline - 30.0:
                        break
                    r = _compute_r(c0, c1, c2)
                    if r is not None and r < best_r:
                        best_r = r
                        best_coeffs = np.array([c0, c1, c2])
    
    # Phase 3: Even finer grid if time permits
    if time.time() < deadline - 30.0:
        c0_center, c1_center, c2_center = best_coeffs
        c0_fine2 = np.linspace(c0_center - 0.1, c0_center + 0.1, 40)
        c1_fine2 = np.linspace(max(0, c1_center - 0.1), c1_center + 0.1, 40)
        c2_fine2 = np.linspace(max(0, c2_center - 0.02), c2_center + 0.02, 30)
        
        for c0 in c0_fine2:
            for c1 in c1_fine2:
                for c2 in c2_fine2:
                    if time.time() > deadline - 15.0:
                        break
                    r = _compute_r(c0, c1, c2)
                    if r is not None and r < best_r:
                        best_r = r
                        best_coeffs = np.array([c0, c1, c2])
    
    return best_coeffs
# EVOLVE-BLOCK-END
