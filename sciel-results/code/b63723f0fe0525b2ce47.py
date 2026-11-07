"""Rewritten program: local search around the seed to minimize the value."""
# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import sys

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

def hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c0, c1, ..., cn] where c_i is coeff of x^i."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    
    H_nm2 = [1]  # H_0
    H_nm1 = [0, 2] # H_1
    
    for k in range(2, n + 1):
        term1 = [0] + [2 * c for c in H_nm1]
        term2 = [-2 * (k - 1) * c for c in H_nm2]
        
        max_len = max(len(term1), len(term2))
        while len(term1) < max_len:
            term1.append(0)
        while len(term2) < max_len:
            term2.append(0)
            
        H_n = [t1 + t2 for t1, t2 in zip(term1, term2)]
        
        H_nm2 = H_nm1
        H_nm1 = H_n
        
    return H_nm1

H0_coeffs = hermite_coeffs(0)
H4_coeffs = hermite_coeffs(4)
H8_coeffs = hermite_coeffs(8)
H12_coeffs = hermite_coeffs(12)

def evaluate_value(c0, c1, c2):
    """
    Compute the value for given coefficients c0, c1, c2.
    Returns the value (float) or None if invalid.
    """
    try:
        c0_f = Fraction(c0)
        c1_f = Fraction(c1)
        c2_f = Fraction(c2)
    except:
        return None

    def add_poly(base, coeffs, scale):
        for i, c in enumerate(coeffs):
            base[i] += scale * c
        return base

    g_coeffs = [Fraction(0) for _ in range(13)]
    add_poly(g_coeffs, H0_coeffs, c0_f)
    add_poly(g_coeffs, H4_coeffs, c1_f)
    add_poly(g_coeffs, H8_coeffs, c2_f)
    
    H0_0 = Fraction(H0_coeffs[0])
    H4_0 = Fraction(H4_coeffs[0])
    H8_0 = Fraction(H8_coeffs[0])
    H12_0 = Fraction(H12_coeffs[0])
    
    if H12_0 == 0:
        return None
    
    c3_f = -(c0_f * H0_0 + c1_f * H4_0 + c2_f * H8_0) / H12_0
    
    add_poly(g_coeffs, H12_coeffs, c3_f)
    
    if g_coeffs[12] < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    if g_coeffs[0] != 0 or g_coeffs[1] != 0:
        return None
    
    q_coeffs = [Fraction(0) for _ in range(11)]
    for i in range(11):
        q_coeffs[i] = g_coeffs[i+2]
    
    # Convert to float for root finding
    q_floats = [float(c) for c in q_coeffs]
    q_reversed = q_floats[::-1]
    
    try:
        roots = np.roots(q_reversed)
    except:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    def q_val(x):
        val = 0
        for c in q_coeffs[::-1]:
            val = val * x + c
        return val
    
    real_roots.sort(reverse=True)
    
    r_best = None
    for r in real_roots:
        # Use a smaller eps for precision, but check multiple points if necessary
        eps = 1e-5
        q_left = q_val(r - eps)
        q_right = q_val(r + eps)
        
        if q_left * q_right < 0:
            r_best = r
            break
    
    if r_best is None:
        return None
    
    value = (r_best ** 2) / (2 * np.pi)
    return value

def construct():
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    if best_val is None:
        return best_c
    
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except:
        has_scipy = False
    
    start_time = time.time()
    deadline = 190.0  # Safety margin for 210s limit
    
    if has_scipy:
        def objective(x):
            val = evaluate_value(x[0], x[1], x[2])
            if val is None:
                return 1e10
            return val
        
        # Run Nelder-Mead with more iterations and tighter tolerance
        # Also try a few different starting points if time permits
        x0 = best_c
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-14})
        
        if time.time() < deadline:
            val_new = evaluate_value(res.x[0], res.x[1], res.x[2])
            if val_new is not None and val_new < best_val:
                best_c = res.x
                best_val = val_new
        
        # Second pass with a slightly perturbed start if time allows
        if time.time() < deadline - 60:
            x0_perturbed = best_c + np.array([1e-3, 1e-3, 1e-5])
            res2 = minimize(objective, x0_perturbed, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-14})
            val_new2 = evaluate_value(res2.x[0], res2.x[1], res2.x[2])
            if val_new2 is not None and val_new2 < best_val:
                best_c = res2.x
                best_val = val_new2

    # Fallback or refinement: Coordinate descent
    step = 1e-4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(3):
            for direction in [1, -1]:
                c_new = best_c.copy()
                c_new[i] += direction * step
                val_new = evaluate_value(c_new[0], c_new[1], c_new[2])
                if val_new is not None and val_new < best_val:
                    best_c = c_new
                    best_val = val_new
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-12:
                break
    
    return best_c
# EVOLVE-BLOCK-END
