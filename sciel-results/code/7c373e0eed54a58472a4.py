"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _laguerre_coeffs(n, alpha):
    """Return coefficients of L_n^(alpha)(x) as [c_n, ..., c_0]."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([1.0 + alpha, -1.0])
    L_prev = np.array([1.0])
    L_curr = np.array([1.0 + alpha, -1.0])
    for k in range(1, n):
        term1 = np.convolve(L_curr, [-1.0, 2.0 * k + alpha + 1.0])
        term2 = (k + alpha) * L_prev
        max_len = max(len(term1), len(term2))
        t1 = np.zeros(max_len); t1[:len(term1)] = term1
        t2 = np.zeros(max_len); t2[:len(term2)] = term2
        L_next = (t1 - t2) / (k + 1.0)
        L_prev, L_curr = L_curr, L_next
    return L_curr

def _compute_score_float(z_list):
    """Fast float approximation of B/(2*pi)."""
    z = np.array(z_list, dtype=float)
    if len(z) != 12 or np.any(z <= 0) or np.any(~np.isfinite(z)):
        return 1e10
    z_sorted = np.sort(z)
    if np.any(np.diff(z_sorted) < 1e-10):
        return 1e10
    
    alpha = -0.5
    points = np.concatenate(([0.0], z_sorted))
    N = len(points)
    
    # Precompute Laguerre values at all points using recurrence
    Lm = np.zeros((51, N))  # L_k^(-1/2)
    Lp = np.zeros((51, N))  # L_k^(1/2) for derivative relation
    
    Lm[0, :] = 1.0
    Lp[0, :] = 1.0
    Lm[1, :] = 0.5 - points
    Lp[1, :] = 1.5 - points
    
    for k in range(1, 50):
        Lm[k+1, :] = ((2*k + 0.5 - points) * Lm[k, :] - (k - 0.5) * Lm[k-1, :]) / (k + 1)
        Lp[k+1, :] = ((2*k + 1.5 - points) * Lp[k, :] - (k + 0.5) * Lp[k-1, :]) / (k + 1)
    
    # Build 26x26 system
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        n = 2 * k
        A[0, k] = Lm[n, 0]
        A[1, k] = Lp[n-1, 0] if n > 0 else 0.0
        for i in range(12):
            idx = 1 + i
            A[2+2*i, k] = Lm[n, idx]
            A[3+2*i, k] = Lp[n-1, idx] if n > 0 else 0.0
    
    b[0] = 0.0
    b[1] = 1.0
    
    try:
        c = np.linalg.solve(A, b)
    except:
        return 1e10
    
    # Build g coefficients
    g_coeffs = np.zeros(51)
    for k in range(26):
        n = 2 * k
        lc = _laguerre_coeffs(n, alpha)
        g_coeffs[:len(lc)] += c[k] * lc
    
    # Denominator: x * prod (x - z_i)^2
    D = np.array([1.0, 0.0])
    for zi in z_sorted:
        D = np.convolve(D, np.array([1.0, -2*zi, zi*zi]))
    
    try:
        q_c, rem = np.polydiv(g_coeffs, D)
    except:
        return 1e10
    
    if len(q_c) == 0:
        return 1e10
    
    roots = np.roots(q_c)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    
    if not real_roots:
        return 1e10
    
    B = max(0.0, max(real_roots))
    return B / (2.0 * math.pi)

def _compute_score_exact(z_list, deadline):
    """Exact rational score computation."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, laguerre as sympy_laguerre
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        try:
            laguerre_pols.append(sympy_laguerre(d, alpha, x))
        except:
            return None
    
    if time.time() > deadline:
        return None
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        rv, rd = 2 + 2*i, 3 + 2*i
        for k in range(26):
            A[rv, k] = laguerre_pols[k].subs(x, zi)
            A[rd, k] = laguerre_pols[k].diff(x).subs(x, zi)
    
    if time.time() > deadline:
        return None
    
    try:
        c = A.LUsolve(b)
    except:
        return None
    
    if time.time() > deadline:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    denom = x
    for zi in z_rationals:
        denom *= (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, _ = g_poly.div(denom_poly)
    except:
        return None
    
    if time.time() > deadline:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        rr = real_roots(q_expr)
    except:
        return None
    
    if time.time() > deadline:
        return None
    
    eps = Rational(1, 10)**198
    max_r = 0
    for r in rr:
        if time.time() > deadline:
            return None
        try:
            if q_expr.subs(x, r - eps) * q_expr.subs(x, r + eps) < 0:
                max_r = max(max_r, r)
        except:
            continue
    
    return float(max_r / (2 * sympy_pi))

def _is_valid(z):
    if len(z) != 12:
        return False
    for v in z:
        if not math.isfinite(v) or v <= 0:
            return False
    s = sorted(z)
    return all(s[i+1] - s[i] > 1e-10 for i in range(11))

def _bessel_zero_J0(k):
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def construct():
    deadline = time.time() + 570.0
    
    # Initial candidates from multiple families
    z_bessel = [_bessel_zero_J0(k)**2 for k in range(1, 13)]
    z_airy = [((k + 0.5) * math.pi / 2)**2 for k in range(1, 13)]
    z_geom = [0.3 * (1.8 ** i) for i in range(12)]
    z_lin = [0.5 + 1.5 * i for i in range(12)]
    
    candidates = [z_bessel, z_airy, z_geom, z_lin]
    valid_candidates = [c for c in candidates if _is_valid(c)]
    
    if not valid_candidates:
        return [0.5 + 1.5 * i for i in range(12)]
    
    # Score all initial candidates
    scored = []
    for c in valid_candidates:
        s = _compute_score_float(c)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0])
    
    best_z = list(scored[0][1])
    best_float = scored[0][0]
    
    # Differential evolution for global search
    if time.time() < deadline - 180:
        def obj(u):
            z = np.exp(u)
            if not _is_valid(z):
                return 1e10
            return _compute_score_float(z)
        
        u0 = np.log(np.array(best_z))
        bounds = [(np.log(0.01), np.log(500.0))] * 12
        
        try:
            res = differential_evolution(
                obj, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-6, polish=False, init='latinhypercube'
            )
            if res.fun < best_float:
                z_opt = np.exp(res.x)
                if _is_valid(z_opt):
                    best_z = list(z_opt)
                    best_float = res.fun
        except:
            pass
    
    # Local refinement with adaptive steps
    if time.time() < deadline - 120:
        for scale in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            if time.time() > deadline - 60:
                break
            improved = False
            for i in range(12):
                if time.time() > deadline - 60:
                    break
                for sign in [1, -1]:
                    cand = list(best_z)
                    cand[i] *= (1 + sign * scale)
                    if not _is_valid(cand):
                        continue
                    s = _compute_score_float(cand)
                    if s < best_float - 1e-12:
                        best_z = cand
                        best_float = s
                        improved = True
            if not improved:
                break
    
    # Pairwise moves
    if time.time() < deadline - 90:
        for i in range(12):
            if time.time() > deadline - 60:
                break
            for j in range(i + 1, 12):
                if time.time() > deadline - 60:
                    break
                for di in [0.02, -0.02, 0.01, -0.01]:
                    for dj in [0.02, -0.02]:
                        if time.time() > deadline - 60:
                            break
                        cand = list(best_z)
                        cand[i] *= (1 + di)
                        cand[j] *= (1 + dj)
                        if not _is_valid(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_float - 1e-12:
                            best_z = cand
                            best_float = s
    
    # Exact evaluation of best and nearby candidates
    exact_results = []
    if time.time() < deadline - 80:
        s = _compute_score_exact(best_z, deadline - 30)
        if s is not None:
            exact_results.append((s, best_z))
    
    # Try exact eval of perturbed versions
    if time.time() < deadline - 60:
        for i in range(12):
            if time.time() > deadline - 45:
                break
            for step in [0.005, 0.002]:
                for sign in [1, -1]:
                    if time.time() > deadline - 30:
                        break
                    cand = list(best_z)
                    cand[i] *= (1 + sign * step)
                    if not _is_valid(cand):
                        continue
                    s = _compute_score_exact(cand, deadline - 20)
                    if s is not None:
                        exact_results.append((s, cand))
    
    if exact_results:
        exact_results.sort(key=lambda x: x[0])
        best_z = exact_results[0][1]
    
    return best_z

# EVOLVE-BLOCK-END
