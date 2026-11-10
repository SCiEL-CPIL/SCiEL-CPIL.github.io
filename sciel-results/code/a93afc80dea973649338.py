"""Refined program: generalized eigenvalue-based root placement with time-budgeted refinement."""
import time
import math

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """High-accuracy k-th positive zero of J_0 using asymptotic expansion."""
    if k == 1:
        return 2.4048255576957728
    if k == 2:
        return 5.5200781102863106
    if k == 3:
        return 8.6537279129110122
    if k == 4:
        return 11.7915344390143598
    if k == 5:
        return 14.9309177084877878
    if k == 6:
        return 18.0710639679109298
    if k == 7:
        return 21.2116366299713778
    if k == 8:
        return 24.3524715308462127
    if k == 9:
        return 27.4934791320444287
    if k == 10:
        return 30.6346064684319602
    if k == 11:
        return 33.7758202136374444
    if k == 12:
        return 36.9171012943291185
    # General asymptotic for larger k
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _eigenvalue_roots():
    """Find 12 positive critical points via generalized eigenvalue problem.
    
    We seek z_1..z_12 as the positive roots of P'(x) where P(x) = prod(x-z_i)
    minimizes the L^2 norm of the residual under Laguerre weight. This is
    equivalent to finding the 12 smallest positive eigenvalues of a generalized
    eigenvalue problem derived from the moment matrix.
    """
    import numpy as np
    from scipy.linalg import eigh
    
    # Build the moment matrix for Laguerre weight w(x) = x^(-1/2) * exp(-x) on [0, inf)
    # M_ij = integral_0^inf x^(i+j) * x^(-1/2) * exp(-x) dx = Gamma(i+j+1/2)
    # We need a finite-dimensional approximation. Use N=30 basis functions.
    N = 30
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = math.gamma(i + j + 0.5)
    
    # The generalized eigenvalue problem: M * v = lambda * D * v
    # where D is a diagonal matrix related to the derivative operator.
    # For the critical points of P(x), we need the eigenvalues of the
    # multiplication-by-x operator in the orthogonal basis.
    
    # Alternative approach: use the three-term recurrence for Laguerre polynomials
    # to build a Jacobi matrix, whose eigenvalues are the zeros of the orthogonal
    # polynomial of degree 12.
    
    # The zeros of L_12^(alpha)(x) for alpha = -1/2 give natural critical points.
    # But we want 12 distinct positive points that work well for the interpolation.
    
    # Let's use the eigenvalue approach: the zeros of the orthogonal polynomial
    # of degree 12 with weight x^(-1/2) exp(-x) are the optimal quadrature points.
    
    # Build the Jacobi matrix for generalized Laguerre polynomials L_n^(alpha)
    # Three-term recurrence: (n+1) L_{n+1}(x) = (2n + alpha + 1 - x) L_n(x) - (n + alpha) L_{n-1}(x)
    # In monic form: x * p_n = p_{n+1} + a_n * p_n + b_n * p_{n-1}
    # where a_n = -(2n + alpha + 1)/(n+1), b_n = (n + alpha)/(n+1) * (n)/(n) ... 
    
    # Actually, let me use scipy's roots_laguerre for the standard case,
    # or build the Jacobi matrix directly.
    
    alpha = -0.5
    Nroots = 12
    
    # Jacobi matrix J for generalized Laguerre L_n^(alpha)
    # The recurrence for monic Laguerre polynomials:
    # p_0 = 1
    # p_1 = x - (1 + alpha)
    # p_{n+1} = (x - (2n + alpha + 1)/(n+1)) * p_n - (n + alpha)/(n+1) * n/(n) * p_{n-1}
    # Wait, let me be more careful.
    
    # Standard three-term recurrence for L_n^(alpha):
    # n * L_n^(alpha) = (2n-1+alpha-x) L_{n-1}^(alpha) - (n-1+alpha) L_{n-2}^(alpha)
    # 
    # For the Jacobi matrix (which gives the zeros), we use:
    # J[i,i] = (2*i + 1 + alpha) / (i + 1)   for i = 0, 1, ..., N-1
    # J[i,i-1] = J[i-1,i] = sqrt(i * (i + alpha)) / (i + 1)  for i = 1, ..., N-1
    
    J = np.zeros((Nroots, Nroots))
    for i in range(Nroots):
        J[i, i] = (2*i + 1 + alpha) / (i + 1)
        if i > 0:
            offdiag = math.sqrt(i * (i + alpha)) / (i + 1)
            J[i, i-1] = offdiag
            J[i-1, i] = offdiag
    
    # The eigenvalues of J are the zeros of L_Nroots^(alpha)(x)
    eigenvalues, _ = eigh(J)
    eigenvalues = np.sort(eigenvalues)
    
    # Filter to positive values
    positive_eigs = [float(e) for e in eigenvalues if e > 1e-10]
    
    if len(positive_eigs) < Nroots:
        # Fallback to Bessel zeros
        return [_bessel_zero_J0(i+1)**2 for i in range(Nroots)]
    
    return positive_eigs[:Nroots]

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, binomial, S)
    from sympy.polys.orthopoly import genlaguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = S(-1)/2

    # Build L_{2k}^{alpha}(x) for k=0..25 using sympy's genlaguerre
    laguerre_pols = []
    for k in range(26):
        n = 2*k
        laguerre_pols.append(genlaguerre(n, alpha)(x))

    # Build system matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: S(0))
    b = Matrix(26, 1, lambda i, j: S(0))

    # Precompute values for speed
    for k in range(26):
        L = laguerre_pols[k]
        Ld = L.diff(x)
        A[0, k] = L.subs(x, 0)
        A[1, k] = Ld.subs(x, 0)
        for i in range(12):
            zi = z_rationals[i]
            A[2 + 2*i, k] = L.subs(x, zi)
            A[3 + 2*i, k] = Ld.subs(x, zi)
    b[1, 0] = 1

    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    # Denominator: x * prod (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2

    g_poly = Poly(g, x)
    denom_poly = Poly(denom)
    try:
        q_poly, rem = g_poly.div(denom_poly)
        if rem != 0:
            return None
    except Exception:
        return None

    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None

    eps = Rational(1, 10)**198
    max_r = S(0)
    for r in real_roots_list:
        try:
            qm = q_expr.subs(x, r - eps)
            qp = q_expr.subs(x, r + eps)
            if qm * qp < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue

    B = max(S(0), max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _is_valid(z_list):
    """Check if z_list is valid: finite, positive, pairwise distinct."""
    for i in range(12):
        if not math.isfinite(z_list[i]) or z_list[i] <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-9:
                return False
    return True

def construct():
    """Return 12 double-root locations using eigenvalue-based construction with refinement."""
    deadline = time.time() + 570.0  # 60s safety margin

    # Primary construction: eigenvalue-based root placement
    try:
        result = _eigenvalue_roots()
    except Exception:
        result = [_bessel_zero_J0(i+1)**2 for i in range(12)]

    # Validate
    if not _is_valid(result):
        result = [_bessel_zero_J0(i+1)**2 for i in range(12)]
        if not _is_valid(result):
            result = [float(i+1) * 5.0 for i in range(12)]

    # Time-budgeted refinement: try geometric scaling and local search
    remaining = deadline - time.time()
    if remaining > 100:
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Try several geometric scalings
                scales = [0.3, 0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0]
                for s in scales:
                    if time.time() > deadline - 120:
                        break
                    candidate = [z * s for z in result]
                    if not _is_valid(candidate):
                        continue
                    score = _compute_score_exact(candidate)
                    if score is not None and score < best_score:
                        best_score = score
                        result = candidate
        except Exception:
            pass

    # If we still have time, try a finer local search around the best result
    remaining = deadline - time.time()
    if remaining > 150:
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Try small perturbations of each z_i
                step = 0.1
                for trial in range(50):
                    if time.time() > deadline - 120:
                        break
                    candidate = list(result)
                    idx = trial % 12
                    direction = 1 if (trial // 12) % 2 == 0 else -1
                    candidate[idx] += direction * step
                    if not _is_valid(candidate):
                        continue
                    score = _compute_score_exact(candidate)
                    if score is not None and score < best_score:
                        best_score = score
                        result = candidate
                        step *= 0.5
                    else:
                        step *= 0.7
        except Exception:
            pass

    return result
# EVOLVE-BLOCK-END
