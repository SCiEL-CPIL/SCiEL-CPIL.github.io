"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    # L_n^(alpha)(x) = sum_{k=0}^{n} binom(n+alpha, n-k) * (-1)^k * x^k / k!
    laguerre_pols = []
    for d in range(0, 51, 2):
        result = 0
        for k in range(d + 1):
            coeff = binomial(d + alpha, d - k) * (-1)**k
            result = result + coeff * x**k / math.factorial(k)
        laguerre_pols.append(result)
    
    # Precompute derivatives
    laguerre_derivs = [p.diff(x) for p in laguerre_pols]
    
    # Build matrix A and vector b
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_derivs[k].subs(x, 0)
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
            A[row_idx, k] = laguerre_derivs[k].subs(x, zi)
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
        q_poly, _ = g_poly.div(denom_poly)
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

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z(z_list):
    """Check that z_list is valid: 12 positive, finite, pairwise distinct floats."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 590.0  # 630s kill, keep 40s margin for return/safety
    
    # Strategy: Use a single high-quality initial candidate to ensure we finish in time.
    # The Bessel J0 zeros squared are a strong heuristic for Laguerre roots.
    
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    if not _validate_z(z_bessel):
        # Fallback valid configuration if Bessel roots fail validation (unlikely)
        return [4.0 * (i + 1) for i in range(12)]
    
    # We only evaluate one candidate to guarantee completion within the time limit.
    # The cost of one exact evaluation is significant (seconds to minute), so multiple
    # evaluations risk timeout.
    
    try:
        # Check time before starting the expensive computation
        if _time_remaining(deadline) > 500:
            score = _compute_score_exact(z_bessel)
            # If score is valid, we return this z_list.
            # Note: We don't actually use the score value to change z_list in this single-shot mode,
            # but we verify it runs without exception.
            if score is not None:
                return z_bessel
        else:
            # Not enough time, return the candidate directly (it is valid)
            return z_bessel
    except Exception:
        # If computation fails or times out internally, return the candidate anyway
        # as it is a valid list of floats.
        return z_bessel
# EVOLVE-BLOCK-END
