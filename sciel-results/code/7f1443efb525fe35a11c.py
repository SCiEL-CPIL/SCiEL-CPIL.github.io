"""Rewritten program for the suite task; stronger algorithm in EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    if k == 1:
        return 2.4048255576957728
    if k == 2:
        return 5.5200781102863106
    if k == 3:
        return 8.653727912911012
    if k == 4:
        return 11.791534439014381
    if k == 5:
        return 14.930917708487787
    if k == 6:
        return 18.071063967910928
    if k == 7:
        return 21.21163662997139
    if k == 8:
        return 24.35247153084626
    if k == 9:
        return 27.493479132041405
    if k == 10:
        return 30.63460646843116
    if k == 11:
        return 33.7758202136327
    if k == 12:
        return 36.91710129412621
    # Asymptotic for larger k
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial
    from sympy import factorial as sfactorial
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # L_n^(alpha)(x) = sum_{k=0}^{n} binom(n+alpha, n-k) * (-x)^k / k!
    laguerre_pols = []
    for d in range(0, 51, 2):
        result = 0
        for k in range(d + 1):
            coeff = binomial(d + alpha, d - k) * (-1)**k
            result = result + coeff * x**k / sfactorial(k)
        laguerre_pols.append(result)
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
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
    
    # Check if remainder is zero (it should be since g has those roots)
    # Actually, the problem says "exact polynomial division" - if there's a remainder, 
    # the setup might be inconsistent, but let's proceed with the quotient
    
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

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 580.0  # Keep 50s margin before 630s kill
    
    # Start with Bessel zeros of J_0 squared - these are natural scales for Laguerre
    # The zeros of J_0 are approximately (k-1/4)*pi, so squared gives ~ (k-1/4)^2 * pi^2
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Ensure positive, finite, distinct
    result = list(z_bessel)
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    # Evaluate the base score
    best_score = _compute_score_exact(result)
    if best_score is None:
        return result
    
    # Try to improve by optimizing the z values
    # Strategy: use scipy.optimize to minimize the score
    # But each evaluation is expensive (seconds to minute), so limit iterations
    
    remaining = _time_remaining(deadline)
    if remaining > 150:  # Need at least ~150s for one evaluation
        try:
            from scipy.optimize import minimize
            
            # Define objective function
            def objective(z_params):
                z_candidate = list(z_params)
                # Check validity
                for i in range(12):
                    if z_candidate[i] <= 0 or not math.isfinite(z_candidate[i]):
                        return 1e10
                for i in range(12):
                    for j in range(i+1, 12):
                        if abs(z_candidate[i] - z_candidate[j]) < 1e-10:
                            return 1e10
                score = _compute_score_exact(z_candidate)
                if score is None:
                    return 1e10
                return score
            
            # Try a few optimization steps
            # Use Nelder-Mead since we don't have gradients
            # Limit function evaluations due to time constraints
            max_evals = max(1, int((remaining - 60) / 60))  # ~60s per eval estimate
            
            if max_evals > 1:
                res = minimize(objective, result, method='Nelder-Mead',
                             options={'maxiter': max_evals, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success and res.fun < best_score:
                    result = list(res.x)
                    best_score = res.fun
        except Exception:
            pass
    
    # Final validation
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    return result
# EVOLVE-BLOCK-END
