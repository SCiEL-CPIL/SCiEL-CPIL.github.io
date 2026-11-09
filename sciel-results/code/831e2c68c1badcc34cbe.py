"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

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
    
    # Strategy: Use Laguerre polynomial zeros as initial candidates
    # The zeros of L_n^(alpha)(x) are the nodes for Gaussian quadrature with weight x^alpha e^{-x}
    # We want 12 positive distinct roots.
    
    # Option 1: Use zeros of L_12^(-1/2)(x) - these are the Gauss-Laguerre nodes
    # Option 2: Use a mix of zeros from different degrees for better spread
    # Option 3: Use zeros of L_12^(alpha)(x) scaled to optimize
    
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial
    import sympy
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / sympy.factorial(k)
        return result
    
    # Get zeros of L_12^(-1/2)(x)
    L12 = L_gen_explicit(12, alpha, x)
    try:
        zeros_L12 = real_roots(L12)
        # Filter positive real zeros
        z_candidates = [float(z) for z in zeros_L12 if z > 0]
    except Exception:
        z_candidates = []
    
    if len(z_candidates) >= 12:
        # Take the 12 smallest positive zeros
        z_candidates.sort()
        z_list = z_candidates[:12]
    else:
        # Fallback: use Bessel-zero-inspired placement
        z_list = [4.0 * (i + 1) for i in range(12)]
    
    # Ensure validity: positive, finite, pairwise distinct
    for i in range(12):
        if not math.isfinite(z_list[i]) or z_list[i] <= 0:
            z_list = [4.0 * (i + 1) for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                z_list = [4.0 * (i + 1) for i in range(12)]
                break
    
    # Evaluate the base score
    best_score = None
    best_z = list(z_list)
    
    if time.time() < deadline - 120:
        try:
            best_score = _compute_score_exact(z_list)
        except Exception:
            best_score = None
    
    # Try to refine by scaling
    if best_score is not None and time.time() < deadline - 90:
        for scale in [0.8, 1.2, 0.6, 1.5, 0.5, 2.0]:
            if time.time() > deadline - 90:
                break
            candidate = [z * scale for z in z_list]
            # Validate
            valid = True
            for i in range(12):
                if not math.isfinite(candidate[i]) or candidate[i] <= 0:
                    valid = False
                    break
            if valid:
                for i in range(12):
                    for j in range(i+1, 12):
                        if abs(candidate[i] - candidate[j]) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                try:
                    s = _compute_score_exact(candidate)
                    if s is not None and (best_score is None or s < best_score):
                        best_score = s
                        best_z = candidate
                except Exception:
                    pass
    
    # Try individual perturbations
    if best_score is not None and time.time() < deadline - 60:
        for i in range(12):
            if time.time() > deadline - 60:
                break
            for delta_frac in [0.05, -0.05, 0.1, -0.1]:
                if time.time() > deadline - 60:
                    break
                candidate = list(best_z)
                candidate[i] = best_z[i] * (1 + delta_frac)
                if candidate[i] <= 0:
                    continue
                valid = True
                for a in range(12):
                    for b_idx in range(a+1, 12):
                        if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                try:
                    s = _compute_score_exact(candidate)
                    if s is not None and s < best_score:
                        best_score = s
                        best_z = candidate
                except Exception:
                    pass
    
    return best_z

# EVOLVE-BLOCK-END
