"""Rewritten program: Bessel-zero-squared roots with a single guarded exact evaluation."""
import time
import math

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
        return 11.791534439014356
    if k == 5:
        return 14.930917708487787
    if k == 6:
        return 18.07106396791092
    if k == 7:
        return 21.21163662997138
    if k == 8:
        return 24.35247153084626
    if k == 9:
        return 27.493479132044446
    if k == 10:
        return 30.63460646843116
    if k == 11:
        return 33.77582021363798
    if k == 12:
        return 36.91709835371005
    x = (k - 0.25) * math.pi
    return x - 1.0 / (8.0 * x) + 9.0 / (128.0 * x**3) - 225.0 / (3072.0 * x**5)


def _compute_score_exact(z_list, deadline):
    """
    Compute B/(2*pi) for given z_list using sympy exact arithmetic.
    Returns float score or None on failure/timeout.
    """
    try:
        from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial
    except Exception:
        return None

    elapsed = time.time() - deadline + 630.0  # approximate elapsed
    if elapsed > 560:
        return None

    z_rationals = sorted(Rational(float(z)) for z in z_list)

    x = symbols('x')
    alpha = Rational(-1, 2)

    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1) ** k
            result = result + coeff * x_sym**k / math.factorial(k)
        return result

    # Precompute Laguerre polynomials and their derivatives
    laguerre_pols = []
    laguerre_ders = []
    for d in range(0, 51, 2):
        lp = L_gen_explicit(d, alpha, x)
        laguerre_pols.append(lp)
        laguerre_ders.append(lp.diff(x))

    if time.time() - deadline + 630.0 > 570:
        return None

    # Build system: 26 equations, 26 unknowns
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)

    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0

    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_ders[k].subs(x, 0)
    b[1, 0] = 1

    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2 * i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2 * i
        for k in range(26):
            A[row_idx, k] = laguerre_ders[k].subs(x, zi)
        b[row_idx, 0] = 0

    if time.time() - deadline + 630.0 > 580:
        return None

    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    if time.time() - deadline + 630.0 > 590:
        return None

    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi) ** 2

    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)

    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None

    if time.time() - deadline + 630.0 > 600:
        return None

    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None

    if time.time() - deadline + 630.0 > 610:
        return None

    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10) ** 198

    max_r = 0
    for r in real_roots_list:
        if time.time() - deadline + 630.0 > 615:
            break
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
    """
    Return 12 double-root locations z_1..z_12.
    Uses squares of Bessel J_0 zeros as the primary candidate.
    Attempts one exact evaluation if time permits, but always returns
    a valid, finite, positive, pairwise-distinct list within 630s.
    """
    deadline = time.time() + 630.0

    # Primary candidate: squares of Bessel J_0 zeros for k=1..12
    z_bessel = [_bessel_zero_J0(i) ** 2 for i in range(1, 13)]

    # Validate: finite, > 0, pairwise distinct
    result = list(z_bessel)
    valid = True
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            valid = False
            break
    if valid:
        for i in range(12):
            for j in range(i + 1, 12):
                if abs(result[i] - result[j]) < 1e-10:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback: simple distinct positive values
        result = [4.0 * (i + 1) for i in range(12)]

    # Attempt one exact evaluation if there's enough time
    elapsed = time.time() - (deadline - 630.0)
    if elapsed < 500:  # Need at least ~130s for evaluation + margin
        try:
            score = _compute_score_exact(result, deadline)
            # If evaluation succeeded, we could try small perturbations,
            # but given the extreme cost, we stick with the primary candidate.
        except Exception:
            pass

    return result

# EVOLVE-BLOCK-END
