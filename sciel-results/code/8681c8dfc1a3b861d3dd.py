"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import roots_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial
    import sympy
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # L_n^(alpha)(x) = sum_{k=0}^{n} binomial(n+alpha, n-k) * (-x)^k / k!
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / sympy.factorial(k)
        return result
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    # Precompute derivatives
    laguerre_pols_deriv = [p.diff(x) for p in laguerre_pols]
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26)
    b = Matrix(26, 1)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols_deriv[k].subs(x, 0)
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
            A[row_idx, k] = laguerre_pols_deriv[k].subs(x, zi)
        b[row_idx, 0] = 0
    
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def construct():
    """Return 12 double-root locations z_1..z_12 using spectral interpolation strategy."""
    deadline = time.time() + 540.0  # Keep 60s margin
    
    # Strategy: Evaluate exact scores for nodes of Laguerre polynomials with varied alpha
    # This aligns with the "Spectral Interpolation with Laguerre Orthogonality Constraints" directive
    # We assume the optimal roots are close to the nodes of L_12^(alpha) for some alpha
    
    alphas_to_try = [
        -0.5,  # Standard
        -0.4,
        -0.6,
        0.0,
        0.5,
        1.0,
        1.5
    ]
    
    best_z = None
    best_score = float('inf')
    
    for alpha_val in alphas_to_try:
        if time.time() > deadline - 60: # Stop if too close to deadline
            break
        
        try:
            # Get nodes for L_12^(alpha)
            z_candidates = roots_genlaguerre(12, alpha_val)
            z_list = [float(z) for z in z_candidates]
            
            # Validate
            if any(z <= 0 or not math.isfinite(z) for z in z_list):
                continue
            # Check distinctness
            z_sorted = sorted(z_list)
            valid = True
            for i in range(11):
                if abs(z_sorted[i] - z_sorted[i+1]) < 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute exact score
            score = _compute_score_exact(z_list)
            
            if score is not None and score < best_score:
                best_score = score
                best_z = z_list
                
        except Exception:
            continue
            
    # Fallback if no valid candidate found
    if best_z is None:
        try:
            z_init = roots_genlaguerre(12, -0.5)
            best_z = [float(z) for z in z_init]
        except Exception:
            best_z = [4.0 * (i + 1) for i in range(12)]
            
    return best_z

# EVOLVE-BLOCK-END
