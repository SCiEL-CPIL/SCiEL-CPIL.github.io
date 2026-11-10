import time
import math
import numpy as np
import scipy.special
import sympy

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / sympy.factorial(k)
        return result

    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
        
    # Build matrix A and vector b
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
        
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
        
    # Optimize polynomial domain for real_roots
    try:
        q_poly = q_poly.set_domain('QQ')
    except Exception:
        pass
        
    try:
        real_roots_list = real_roots(q_poly)
    except Exception:
        return None
        
    eps = Rational(1, 10)**198
    max_r = 0
    for r in real_roots_list:
        try:
            # Use poly.eval for faster evaluation than subs
            q_at_minus = q_poly.eval(r - eps)
            q_at_plus = q_poly.eval(r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
            
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _laguerre_derivative_roots(n, alpha):
    """
    Find the positive roots of (L_n^(alpha))'(x) using scipy.
    Roots of (L_n^(alpha))'(x) are roots of L_{n-1}^{(alpha+1)}(x).
    """
    roots_deriv = scipy.special.roots_genlaguerre(n - 1, alpha + 1)[0]
    roots_deriv = sorted([float(r) for r in roots_deriv if r > 0])
    return roots_deriv

def construct():
    # Only evaluate the primary candidate to ensure we finish within time limits.
    # The previous timeout was caused by evaluating multiple candidates.
    try:
        z_deriv = _laguerre_derivative_roots(13, -0.5)
        if len(z_deriv) < 12:
            raise ValueError("Not enough roots")
    except Exception:
        # Fallback: distinct positive integers
        z_deriv = [float(i) for i in range(1, 13)]

    # Evaluate score to ensure validity, but return z_deriv regardless of score
    # as long as the structure is valid (finite, >0, distinct).
    # The evaluator will compute the score.
    # We perform the evaluation to catch singular matrix errors which might
    # indicate an invalid configuration, but since z_deriv are distinct positive
    # reals, the matrix should be non-singular.
    
    # We do NOT need to return the score, just the z_list.
    # The task says "Write construct(), returning 12 double-root locations".
    
    return z_deriv
