"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic.
    Returns None if invalid or timeout."""
    from sympy import (symbols, Rational, Poly, diff, real_roots, pi as sympy_pi,
                       Matrix, laguerre as sympy_laguerre)
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    # Use sympy's built-in laguerre for speed
    laguerre_pols = []
    for d in range(0, 51, 2):
        try:
            lp = sympy_laguerre(d, alpha, x)
            laguerre_pols.append(lp)
        except Exception:
            return None
    
    if time.time() > deadline:
        return None
    
    # Build matrix A (26x26) and vector b (26x1)
    # Row 0: g(0) = 0
    # Row 1: g'(0) = 1
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0 for i=0..11
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Precompute L_{2k}(0) and L'_{2k}(0)
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    # For each z_i
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = laguerre_pols[k].subs(x, zi)
            A[row_der, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    # Solve the linear system
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Build g(x) = sum c_k * L_{2k}(x)
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
    
    if time.time() > deadline:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
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


def _is_valid_z(z_list):
    """Check if z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return False
    return True


def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 560.0  # 70s safety margin before 630s kill
    
    # Strategy: Try multiple initial configurations, pick the best, then refine
    # Key insight: We want B (largest real root of q) to be as small as possible.
    # Bessel zeros squared give B ~ 280. We need much better configurations.
    
    # Try several families of initial z values
    candidates = []
    
    # Family 1: Bessel J0 zeros squared (baseline)
    z_bessel = []
    for k in range(1, 13):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            j0 = 2.4048255576957728
        else:
            j0 = x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        z_bessel.append(j0**2)
    if _is_valid_z(z_bessel):
        candidates.append(z_bessel)
    
    # Family 2: Geometric progression
    for ratio in [1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]:
        z_geo = [0.5 * (ratio ** i) for i in range(12)]
        if _is_valid_z(z_geo):
            candidates.append(z_geo)
    
    # Family 3: Exponential with different base
    for base in [1.2, 1.3, 1.4, 1.5, 1.7, 2.0]:
        z_exp = [0.3 * (base ** i) for i in range(12)]
        if _is_valid_z(z_exp):
            candidates.append(z_exp)
    
    # Family 4: Spaced out more aggressively
    for start in [0.5, 1.0, 2.0]:
        z_spaced = [start + 3.0 * i for i in range(12)]
        if _is_valid_z(z_spaced):
            candidates.append(z_spaced)
    
    # Family 5: Mix of small and large values
    z_mixed = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    if _is_valid_z(z_mixed):
        candidates.append(z_mixed)
    
    # Family 6: Near-equally spaced but with slight variation
    z_var = [1.0 + 0.5*i + 0.1*math.sin(i) for i in range(12)]
    if _is_valid_z(z_var):
        candidates.append(z_var)
    
    # Evaluate all candidates and pick the best
    best_score = float('inf')
    best_z = z_bessel  # fallback
    
    for z_cand in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score = _compute_score_exact(z_cand, deadline - 10)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(z_cand)
        except Exception:
            continue
    
    if time.time() > deadline - 60:
        return best_z
    
    # Local refinement: perturb each z_i and see if score improves
    current_z = list(best_z)
    current_score = best_score
    
    # Adaptive step sizes
    step_sizes = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(12):
            if time.time() > deadline - 30:
                break
            for step in step_sizes:
                for sign in [1, -1]:
                    if time.time() > deadline - 20:
                        break
                    candidate = list(current_z)
                    candidate[i] = current_z[i] * (1 + sign * step)
                    if candidate[i] <= 0:
                        continue
                    if not _is_valid_z(candidate):
                        continue
                    try:
                        score = _compute_score_exact(candidate, deadline - 10)
                        if score is not None and score < current_score:
                            current_score = score
                            current_z = candidate
                            improved = True
                    except Exception:
                        continue
    
    return current_z

# EVOLVE-BLOCK-END
