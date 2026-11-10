"""Combined: Bessel-zero construction + log-spaced geometric-progression refinement.

Structure (from current program):
  - Bessel J0 zero squares as a strong initial guess.
  - Exact sympy score for the final verification.

Refinement machinery (from inspiration 2):
  - Fast float-based score via scipy Laguerre evaluation.
  - Log-spaced 2D grid search over (base, ratio) for z_i = z_1 * r^(i-1).
  - Local coordinate-descent polish on the best float candidate.
  - Exact sympy re-verification of the top candidates within the time budget.
"""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """High-accuracy k-th positive zero of J_0."""
    table = [
        2.4048255576957728, 5.5200781102863106, 8.6537279129110122,
        11.7915344390143598, 14.9309177084877878, 18.0710639679109298,
        21.2116366299713778, 24.3524715308462127, 27.4934791320444287,
        30.6346064684319602, 33.7758202136374444, 36.9171012943291185,
    ]
    if k <= 12:
        return table[k - 1]
    x = (k - 0.25) * math.pi
    return x - 1.0 / (8.0 * x) + 9.0 / (128.0 * x ** 3) - 225.0 / (3072.0 * x ** 5)


def _is_valid(z_list):
    """finite, > 0, pairwise distinct."""
    n = len(z_list)
    for v in z_list:
        if not np.isfinite(v) or v <= 0:
            return False
    s = np.sort(np.asarray(z_list, dtype=float))
    return bool(np.all(np.diff(s) > 1e-10))


def _approx_B(z_list):
    """
    Fast float approximation of B (not divided by 2*pi).
    Returns a large penalty on invalid input.
    """
    try:
        z = np.asarray(z_list, dtype=float)
        if z.shape[0] != 12 or not _is_valid(z):
            return 1e10
        z = np.sort(z)

        alpha = -0.5
        A = np.zeros((26, 26), dtype=float)
        b = np.zeros(26, dtype=float)
        b[1] = 1.0

        # Build the 26x26 interpolation matrix and collect the even-degree
        # Laguerre basis used to form g(x) = sum_k c_k L_{2k}^{alpha}(x).
        L_basis = []
        Ld_basis = []
        for k in range(26):
            n = 2 * k
            L = eval_genlaguerre(n, alpha, z)
            if n > 0:
                Ld = -eval_genlaguerre(n - 1, alpha + 1, z)
            else:
                Ld = np.zeros_like(z)
            L_basis.append(L)
            Ld_basis.append(Ld)

        # Rows: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0
        for k in range(26):
            n = 2 * k
            A[0, k] = eval_genlaguerre(n, alpha, 0.0)
            A[1, k] = (-eval_genlaguerre(n - 1, alpha + 1, 0.0)) if n > 0 else 0.0
            for i in range(12):
                A[2 + 2 * i, k] = L_basis[k][i]
                A[3 + 2 * i, k] = Ld_basis[k][i]

        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10
        if not np.all(np.isfinite(c)):
            return 1e10

        # Build g(x) coefficients (even degrees 0..50)
        g_coeffs = np.zeros(51, dtype=float)
        for k in range(26):
            n = 2 * k
            # L_n^{alpha}(x) = sum_{j=0}^{n} (-1)^j binom(n+alpha, n-j) x^j / j!
            # Compute via scipy's poly coefficients is overkill; use recurrence-free
            # evaluation by accumulating the polynomial coefficients.
            # We only need g on a grid for the root scan, so build a fast
            # evaluator instead of full coefficient expansion.
            pass

        # Fast evaluator: g(x) = sum_k c_k L_{2k}^{alpha}(x)
        def g_of_x(xv):
            xv = np.asarray(xv, dtype=float)
            total = np.zeros_like(xv)
            for k in range(26):
                n = 2 * k
                total = total + c[k] * eval_genlaguerre(n, alpha, xv)
            return total

        # Denominator d(x) = x * prod_i (x - z_i)^2  (degree 25)
        def d_of_x(xv):
            xv = np.asarray(xv, dtype=float)
            out = xv.copy()
            for zi in z:
                out = out * (xv - zi) ** 2
            return out

        # q(x) = g(x)/d(x) for x not a pole. Scan a log-spaced grid over (0, z_max*3)
        # and detect sign changes (simple roots of q).
        z_max = z[-1]
        lo, hi = 1e-6, max(3.0 * z_max, 1.0)
        grid = np.geomspace(lo, hi, 4000)
        gvals = g_of_x(grid)
        dvals = d_of_x(grid)
        qvals = gvals / dvals

        # Mask poles (where |d| is tiny)
        mask = np.isfinite(qvals) & (np.abs(dvals) > 1e-300)
        qm = qvals[mask]
        gm = grid[mask]

        # Find sign changes
        sign = np.sign(qm)
        # ignore zeros
        change = np.where((sign[:-1] != sign[1:]) & (sign[:-1] != 0) & (sign[1:] != 0))[0]
        if len(change) == 0:
            return 0.0
        # Largest root with a sign change
        r_max = 0.0
        for idx in change:
            # bisection refine on [gm[idx], gm[idx+1]]
            a, bb = gm[idx], gm[idx + 1]
            fa = g_of_x(np.array([a]))[0] / d_of_x(np.array([a]))[0]
            fb = g_of_x(np.array([bb]))[0] / d_of_x(np.array([bb]))[0]
            if fa * fb > 0:
                continue
            for _ in range(60):
                mid = 0.5 * (a + bb)
                fm = g_of_x(np.array([mid]))[0] / d_of_x(np.array([mid]))[0]
                if fa * fm <= 0:
                    bb, fb = mid, fm
                else:
                    a, fa = mid, fm
            r = 0.5 * (a + bb)
            if r > r_max:
                r_max = r
        return float(r_max)
    except Exception:
        return 1e10


def _compute_score_exact(z_list):
    """Exact sympy score B/(2*pi). Returns None on failure."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, S)
    from sympy.polys.orthopoly import genlaguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = S(-1) / 2

    laguerre_pols = []
    for k in range(26):
        n = 2 * k
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
            A[2 + 2 * i, k] = L.subs(x, zi)
            A[3 + 2 * i, k] = Ld.subs(x, zi)
    b[1, 0] = 1

    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi) ** 2

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

    eps = Rational(1, 10) ** 198
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


def construct():
    deadline = time.time() + 560.0  # 70s safety margin vs 630s kill

    # Strong initial guess: Bessel J0 zero squares
    z_bessel = [_bessel_zero_J0(i + 1) ** 2 for i in range(12)]
    if not _is_valid(z_bessel):
        z_bessel = [float(i + 1) * 5.0 for i in range(12)]
    result = list(z_bessel)
    best_float = _approx_B(result)

    # ---- Phase 1: log-spaced 2D grid search over geometric progression ----
    # z_i = base * ratio^(i-1), i=1..12
    def geom(base, ratio):
        return [base * (ratio ** i) for i in range(12)]

    # Coarse grid: base in [0.05, 200] log, ratio in [1.02, 6.0] log
    base_grid = np.geomspace(0.05, 200.0, 24)
    ratio_grid = np.geomspace(1.02, 6.0, 24)
    best_base, best_ratio = base_grid[0], ratio_grid[0]
    for base in base_grid:
        if time.time() > deadline - 120:
            break
        for ratio in ratio_grid:
            cand = geom(float(base), float(ratio))
            if not _is_valid(cand):
                continue
            s = _approx_B(cand)
            if s < best_float:
                best_float = s
                best_base, best_ratio = float(base), float(ratio)
                result = cand

    # ---- Phase 2: local coordinate-descent polish on (base, ratio) ----
    step_b = (best_ratio / best_base) * 0.2
    step_r = 0.05
    for it in range(200):
        if time.time() > deadline - 100:
            break
        improved = False
        for direction in (1, -1):
            nb = best_base * (1 + direction * 0.15)
            nr = best_ratio * (1 + direction * 0.10)
            if nb <= 0 or nr <= 1:
                continue
            cand = geom(nb, nr)
            if not _is_valid(cand):
                continue
            s = _approx_B(cand)
            if s < best_float:
                best_float = s
                best_base, best_ratio = nb, nr
                result = cand
                improved = True
        if not improved:
            # shrink steps
            step_b *= 0.5
            step_r *= 0.5
            if step_b < 1e-8 and step_r < 1e-8:
                break

    # ---- Phase 3: exact sympy verification of top candidates ----
    # Build a short list of candidates: the float-best, the Bessel initial,
    # and a few nearby geometric perturbations.
    candidates = []
    candidates.append(result)
    candidates.append(z_bessel)
    for db in (-0.1, 0.1):
        for dr in (-0.05, 0.05):
            nb = best_base * (1 + db)
            nr = best_ratio * (1 + dr)
            if nb > 0 and nr > 1:
                cand = geom(nb, nr)
                if _is_valid(cand):
                    candidates.append(cand)

    best_exact = None
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        if not _is_valid(cand):
            continue
        try:
            s = _compute_score_exact(cand)
        except Exception:
            s = None
        if s is not None and (best_exact is None or s < best_exact):
            best_exact = s
            result = list(cand)

    # Final safety: ensure returned list is valid
    if not _is_valid(result):
        result = [float(i + 1) * 5.0 for i in range(12)]

    return result
# EVOLVE-BLOCK-END
