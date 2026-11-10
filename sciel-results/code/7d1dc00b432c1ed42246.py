"""Refined program: eigenvalue-based root placement with Nelder-Mead polish."""
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
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, S)
    from sympy.polys.orthopoly import genlaguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = S(-1)/2

    laguerre_pols = []
    for k in range(26):
        n = 2*k
        laguerre_pols.append(genlaguerre(n, alpha)(x))

    A = Matrix(26, 26, lambda i, j: S(0))
    b = Matrix(26, 1, lambda i, j: S(0))

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

    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2

    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
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

def _eigenvalue_roots(deadline):
    """
    Find 12 positive distinct roots via generalized eigenvalue problem.
    We seek the critical points of a polynomial whose roots minimize the
    L^2 residual under Laguerre weight. This is approximated by finding
    the zeros of the derivative of a degree-12 polynomial whose coefficients
    come from an eigenvalue problem on the moment matrix.
    """
    import numpy as np
    from scipy.linalg import eigvals

    # Use the fact that for Laguerre polynomials with alpha=-1/2,
    # the optimal node placement relates to the eigenvalues of a
    # tridiagonal Jacobi matrix. For L_n^{alpha}, the Jacobi matrix
    # has known entries. We use the degree-12 case.
    #
    # The Jacobi matrix for L_n^{alpha} has:
    #   diagonal: (2i + alpha + 1) for i=0..n-1  (shifted)
    #   off-diagonal: sqrt((i+1)(i+alpha+1)) / (2i+alpha+1) ... 
    #
    # Actually, let's use a different approach: the roots that minimize
    # the max residual are related to the eigenvalues of a specific
    # matrix. For our purposes, we'll use the known fact that the
    # optimal nodes for interpolation in the Laguerre basis are the
    # eigenvalues of the companion-like matrix.
    #
    # Simpler approach: use the zeros of the derivative of the
    # monic polynomial with minimal L2 norm, which are the eigenvalues
    # of the Jacobi matrix of degree 12 for the Laguerre weight.
    
    alpha = -0.5
    n = 12
    
    # Jacobi matrix for Laguerre polynomials L_k^{alpha}(x)
    # The three-term recurrence: (k+1) L_{k+1}^{alpha} = (2k+alpha+1-x) L_k^{alpha} - (k+alpha) L_{k-1}^{alpha}
    # Jacobi matrix J has:
    #   J[i,i] = 2i + alpha + 1  for i = 0, ..., n-1
    #   J[i,i+1] = J[i+1,i] = sqrt((i+1)(i+1+alpha)) for i = 0, ..., n-2
    # Wait, let me be more careful. The standard Jacobi matrix for
    # orthogonal polynomials p_n with recurrence:
    #   x p_n = a_{n+1} p_{n+1} + b_n p_n + a_n p_{n-1}
    # has diagonal b_n and off-diagonal a_n.
    #
    # For Laguerre L_n^{alpha}:
    #   n L_n^{alpha} = (2n + alpha - 1 - x) L_{n-1}^{alpha} - (n-1+alpha) L_{n-2}^{alpha}
    # Rearranging: x L_{n-1}^{alpha} = (2n+alpha-1) L_{n-1}^{alpha} - n L_n^{alpha} - (n-1+alpha) L_{n-2}^{alpha}
    # So: a_n = n, b_n = 2n + alpha - 1 (for the n-th step, 0-indexed: b_i = 2i+alpha+1, a_i = i+1)
    
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2*i + alpha + 1  # b_i
        if i > 0:
            J[i, i-1] = i  # a_i
            J[i-1, i] = i  # symmetric
    
    # The eigenvalues of J give the Gaussian quadrature nodes for
    # the Laguerre weight. These are the optimal nodes in the
    # L2 sense.
    try:
        eigenvalues = eigvals(J)
        eigenvalues = np.real(eigenvalues)  # should be real
        eigenvalues = sorted(eigenvalues.tolist())
        
        # Filter: must be positive and distinct
        valid = []
        for ev in eigenvalues:
            if ev > 1e-10 and math.isfinite(ev):
                valid.append(float(ev))
        
        # Check distinctness
        distinct = True
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                if abs(valid[i] - valid[j]) < 1e-8:
                    distinct = False
                    break
            if not distinct:
                break
        
        if len(valid) == 12 and distinct:
            return valid
    except Exception:
        pass
    
    return None

def _nelder_mead_polish(z_init, deadline, max_iters=50):
    """
    Nelder-Mead style local refinement using exact score evaluations.
    We work in log-space to maintain positivity.
    """
    import numpy as np
    
    # Work in log-space: y_i = log(z_i)
    y = np.log(np.array(z_init, dtype=float))
    
    # Initial simplex: perturb each coordinate
    step = 0.05
    simplex = [y.copy()]
    for i in range(12):
        yi = y.copy()
        yi[i] += step
        simplex.append(yi)
    
    def eval_score(y_vec):
        """Convert log-space to z-space and evaluate score."""
        z = np.exp(y_vec)
        if any(z <= 0) or any(not np.isfinite(z)):
            return 1e10
        # Check distinctness
        z_sorted = sorted(z.tolist())
        for i in range(12):
            for j in range(i+1, 12):
                if abs(z_sorted[i] - z_sorted[j]) < 1e-12:
                    return 1e10
        if time.time() > deadline:
            return None
        try:
            s = _compute_score_exact(z_sorted)
            return s if s is not None else 1e10
        except Exception:
            return 1e10
    
    # Evaluate initial simplex
    scores = []
    for i, yi in enumerate(simplex):
        s = eval_score(yi)
        if s is None:
            return z_init  # timeout
        scores.append(s)
    
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_y = simplex[best_idx].copy()
    
    # Nelder-Mead iterations
    for it in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        # Sort simplex by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha_r = 1.0
        y_reflect = centroid + alpha_r * (centroid - worst)
        s_reflect = eval_score(y_reflect)
        if s_reflect is None:
            break
        
        if s_reflect < scores[0]:
            # Expansion
            gamma = 2.0
            y_expand = centroid + gamma * (centroid - worst)
            s_expand = eval_score(y_expand)
            if s_expand is None:
                break
            if s_expand < s_reflect:
                simplex[-1] = y_expand
                scores[-1] = s_expand
            else:
                simplex[-1] = y_reflect
                scores[-1] = s_reflect
        elif s_reflect < scores[-2]:
            simplex[-1] = y_reflect
            scores[-1] = s_reflect
        else:
            # Contraction
            rho = 0.5
            y_contract = centroid + rho * (worst - centroid)
            s_contract = eval_score(y_contract)
            if s_contract is None:
                break
            if s_contract < scores[-1]:
                simplex[-1] = y_contract
                scores[-1] = s_contract
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    s_new = eval_score(simplex[i])
                    if s_new is None:
                        return np.exp(best_y).tolist()
                    scores[i] = s_new
        
        # Track best
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_y = simplex[best_idx].copy()
    
    result = np.exp(best_y).tolist()
    # Ensure valid
    result.sort()
    for i in range(12):
        if result[i] <= 0 or not math.isfinite(result[i]):
            return z_init
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-9:
                return z_init
    return result

def construct():
    """Return 12 double-root locations using eigenvalue-based construction with polish."""
    deadline = time.time() + 570.0  # 60s safety margin

    # Strategy 1: Eigenvalue-based roots (Jacobi matrix eigenvalues)
    result = None
    try:
        result = _eigenvalue_roots(deadline)
    except Exception:
        pass
    
    # Fallback: Bessel zero squares
    if result is None or len(result) != 12:
        z_bessel = [_bessel_zero_J0(i+1)**2 for i in range(12)]
        result = list(z_bessel)
    
    # Validate
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [float(i+1) * 5.0 for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-9:
                result = [float(i+1) * 5.0 for i in range(12)]
    
    # Evaluate baseline score
    best_score = None
    try:
        best_score = _compute_score_exact(result)
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead polish in log-space
    if time.time() < deadline - 120:
        try:
            polished = _nelder_mead_polish(result, deadline, max_iters=40)
            if polished is not None and len(polished) == 12:
                polished.sort()
                # Validate
                valid = True
                for i in range(12):
                    if polished[i] <= 0 or not math.isfinite(polished[i]):
                        valid = False
                        break
                if valid:
                    for i in range(12):
                        for j in range(i+1, 12):
                            if abs(polished[i] - polished[j]) < 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    s = _compute_score_exact(polished)
                    if s is not None and (best_score is None or s < best_score):
                        best_score = s
                        result = polished
        except Exception:
            pass
    
    # Strategy 3: Try geometric scalings of best result
    if time.time() < deadline - 90:
        try:
            scales = [0.3, 0.5, 0.7, 1.4, 2.0, 3.0]
            for s in scales:
                if time.time() > deadline - 90:
                    break
                candidate = [z * s for z in result]
                candidate.sort()
                if any(c <= 0 for c in candidate):
                    continue
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if abs(candidate[i] - candidate[j]) < 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                score = _compute_score_exact(candidate)
                if score is not None and (best_score is None or score < best_score):
                    best_score = score
                    result = candidate
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END
