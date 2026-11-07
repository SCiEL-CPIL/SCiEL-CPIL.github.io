"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import sympy as sp
import time

# Precompute Hermite polynomial coefficients using sympy for exactness
# Physicists' Hermite polynomials
def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as a list [c0, c1, ..., cn] for x^0, x^1, ..."""
    x = sp.Symbol('x')
    Hn = sp.hermite(n, x)
    poly = sp.Poly(Hn, x)
    # Return coefficients from highest degree to lowest (numpy convention)
    return np.array(poly.all_coeffs(), dtype=np.float64)

# Precompute H0, H4, H8, H12 coefficients
_H0_coeffs = _get_hermite_coeffs(0)   # [1]
_H4_coeffs = _get_hermite_coeffs(4)   # degree 4
_H8_coeffs = _get_hermite_coeffs(8)   # degree 8
_H12_coeffs = _get_hermite_coeffs(12) # degree 12

def _build_g_and_q(c0, c1, c2):
    """
    Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12 with g(0)=0.
    Return q(x) = g(x)/x^2 as a polynomial (numpy array, highest degree first).
    If g tends to negative limit at infinity, return -q.
    """
    # g is degree 12 polynomial. Coefficients from x^12 down to x^0.
    # H0 has degree 0, H4 degree 4, H8 degree 8, H12 degree 12.
    # Build coefficient array of length 13 (degrees 12 down to 0)
    g_coeffs = np.zeros(13, dtype=np.float64)
    
    # H0_coeffs: [1] -> degree 0
    # H4_coeffs: 5 coeffs -> degrees 4,3,2,1,0
    # H8_coeffs: 9 coeffs -> degrees 8,7,6,5,4,3,2,1,0
    # H12_coeffs: 13 coeffs -> degrees 12,11,10,9,8,7,6,5,4,3,2,1,0
    
    def add_poly(coeffs, multiplier, g_arr):
        """Add multiplier * poly to g_arr. coeffs is highest-degree first."""
        deg = len(coeffs) - 1
        offset = 12 - deg  # how many zeros to prepend
        for i, c in enumerate(coeffs):
            g_arr[offset + i] += multiplier * c
    
    g_arr = np.zeros(13, dtype=np.float64)
    add_poly(_H0_coeffs, c0, g_arr)
    add_poly(_H4_coeffs, c1, g_arr)
    add_poly(_H8_coeffs, c2, g_arr)
    
    # g(0) = g_arr[12] (constant term, last element)
    # H12(0) = _H12_coeffs[-1]
    H12_0 = _H12_coeffs[-1]
    c3 = -(c0 * _H0_coeffs[-1] + c1 * _H4_coeffs[-1] + c2 * _H8_coeffs[-1]) / H12_0
    add_poly(_H12_coeffs, c3, g_arr)
    
    # Now g_arr[0] is x^12 coeff, g_arr[12] is x^0 coeff
    # g(0) should be ~0
    # q(x) = g(x) / x^2: divide by x^2
    # q_coeffs[i] = g_coeffs[i+2] for i=0..10
    q_coeffs = g_arr[:11].copy()  # x^10 down to x^0
    
    # Check leading coefficient sign: if g tends to negative limit at infinity, flip
    # Leading coeff of g is g_arr[0]
    if g_arr[0] < 0:
        q_coeffs = -q_coeffs
    
    return q_coeffs

def _find_largest_sign_changing_root(q_coeffs):
    """
    Find the largest real root of q at which q changes sign.
    q_coeffs is numpy array, highest degree first.
    """
    # Use numpy roots
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # For each real root, check if q changes sign
    # q changes sign at root r if the multiplicity is odd
    # Numerically: evaluate q at r - eps and r + eps
    eps = 1e-8
    best_r = None
    for r in real_roots:
        if r < 0:
            continue  # we want largest, and by symmetry if r is root, -r is too for even poly
        # Check sign change
        q_left = np.polyval(q_coeffs, r - eps)
        q_right = np.polyval(q_coeffs, r + eps)
        if q_left * q_right < 0:
            if best_r is None or r > best_r:
                best_r = r
    
    return best_r

def _compute_value(c0, c1, c2):
    """Compute r^2 / (2*pi) for given coefficients."""
    q_coeffs = _build_g_and_q(c0, c1, c2)
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None:
        return np.inf
    return r**2 / (2 * np.pi)

def _objective(params):
    """Objective function for scipy.optimize.minimize."""
    c0, c1, c2 = params
    val = _compute_value(c0, c1, c2)
    if np.isinf(val) or np.isnan(val):
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, keep safety margin
    
    # Seed point
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    seed_value = _compute_value(*seed)
    best_params = seed.copy()
    best_value = seed_value
    
    # Try multiple optimization methods and starting points
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    # Starting points: seed and some perturbations
    rng = np.random.RandomState(42)
    start_points = [seed.copy()]
    for _ in range(5):
        perturbation = rng.normal(0, 0.1, size=3)
        start_points.append(seed + perturbation)
    
    for method in methods:
        for sp in start_points:
            if time.time() > deadline - 30:  # safety margin
                break
            try:
                result = minimize(_objective, sp, method=method, 
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_value:
                    best_value = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        if time.time() > deadline - 30:
            break
    
    # Final verification
    final_value = _compute_value(*best_params)
    if not np.isfinite(final_value) or final_value > seed_value:
        return np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
