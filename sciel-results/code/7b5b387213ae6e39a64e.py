"""Refined program: geometric progression root placement with local polish."""
import time
import math

# EVOLVE-BLOCK-START

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

def _geometric_roots(z0, r):
    """Generate 12 roots from geometric progression z0 * r^(i-1)."""
    z_list = []
    for i in range(12):
        z = z0 * (r ** i)
        z_list.append(z)
    return z_list

def _local_polish(z_init, deadline, max_iters=30):
    """
    Local refinement: perturb each z_i slightly and keep improvements.
    Uses coordinate-wise search in log-space.
    """
    import numpy as np
    
    z = np.array(z_init, dtype=float)
    y = np.log(z)
    best_score = None
    try:
        best_score = _compute_score_exact(sorted(z.tolist()))
    except Exception:
        return z_init
    
    if best_score is None:
        return z_init
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for step in step_sizes:
        if time.time() > deadline - 30:
            break
        improved = True
        while improved:
            improved = False
            for i in range(12):
                if time.time() > deadline - 30:
                    return np.exp(y).tolist()
                for direction in [-1, 1]:
                    y_trial = y.copy()
                    y_trial[i] += direction * step
                    z_trial = np.exp(y_trial)
                    if any(z_trial <= 0) or any(not np.isfinite(z_trial)):
                        continue
                    z_sorted = sorted(z_trial.tolist())
                    # Check distinctness
                    valid = True
                    for a in range(12):
                        for b in range(a+1, 12):
                            if abs(z_sorted[a] - z_sorted[b]) < 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    try:
                        s = _compute_score_exact(z_sorted)
                        if s is not None and s < best_score:
                            best_score = s
                            y = y_trial
                            improved = True
                    except Exception:
                        pass
    
    result = np.exp(y).tolist()
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
    """Return 12 double-root locations using geometric progression with local polish."""
    deadline = time.time() + 570.0  # 60s safety margin

    # Strategy: Search over geometric progression parameters (z0, r)
    # z_i = z0 * r^(i-1) for i=1..12
    # We want small z0 and r close to 1 to keep roots clustered at small values
    
    best_score = None
    best_result = None
    
    # Search grid: z0 in log-space, r in log-space
    # z0: try values from 0.01 to 10
    # r: try values from 1.01 to 1.5 (close to 1 means clustered)
    
    z0_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    r_values = [1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]
    
    for z0 in z0_values:
        if time.time() > deadline - 120:
            break
        for r in r_values:
            if time.time() > deadline - 120:
                break
            z_list = _geometric_roots(z0, r)
            # Validate
            valid = True
            for i in range(12):
                if z_list[i] <= 0 or not math.isfinite(z_list[i]):
                    valid = False
                    break
            if not valid:
                continue
            z_sorted = sorted(z_list)
            for i in range(12):
                for j in range(i+1, 12):
                    if abs(z_sorted[i] - z_sorted[j]) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            try:
                s = _compute_score_exact(z_sorted)
                if s is not None and (best_score is None or s < best_score):
                    best_score = s
                    best_result = z_sorted
            except Exception:
                pass
    
    # Fallback if no valid result found
    if best_result is None:
        best_result = [float(i+1) * 0.1 for i in range(12)]
        best_score = None
        try:
            best_score = _compute_score_exact(sorted(best_result))
        except Exception:
            pass
    
    # Local polish
    if time.time() < deadline - 60 and best_score is not None:
        try:
            polished = _local_polish(best_result, deadline, max_iters=20)
            if polished is not None and len(polished) == 12:
                polished.sort()
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
                    if s is not None and s < best_score:
                        best_score = s
                        best_result = polished
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
