"""Refined Cohn-Goncalves Laguerre construction: geometric-root optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """Coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = (math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1)
                       - math.lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_norm_sq(poly):
    """||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx  (exact)."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    weights = np.array([math.factorial(j) for j in range(1, max_k + 2)], dtype=float)
    return float(np.dot(sq, weights))


def _build_poly(roots, m=12):
    """psi(x) = L_m^1(x) * prod_i (1 - x/r_i)^2, as ascending coefficients."""
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    return poly


def _c4_bound(roots, m=12):
    """Upper bound on C4 for given positive distinct roots (lower is better)."""
    if _time_left() < 2:
        return float('inf')
    if len(roots) != 12 or any(r <= 0 for r in roots):
        return float('inf')
    if len(set(round(float(r), 9) for r in roots)) != 12:
        return float('inf')
    poly = _build_poly(roots, m)
    if not np.all(np.isfinite(poly)):
        return float('inf')
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    return 1.0 / norm_sq


def _geo_roots(r0, q, n_roots=12):
    """Geometric roots r0*q^i for i=0..n_roots-1."""
    return [float(r0) * (float(q) ** i) for i in range(n_roots)]


def _objective(v, m=12):
    """Objective to minimize: -log(norm_sq). v = [log(r0), log(q)]."""
    if _time_left() < 1:
        return 1e100
    r0 = math.exp(v[0])
    q = math.exp(v[1])
    if not (np.isfinite(r0) and np.isfinite(q)) or r0 <= 0 or q <= 0:
        return 1e100
    if q > 1e4:  # keep roots from overflowing
        return 1e100
    roots = _geo_roots(r0, q)
    if any(r > 1e15 for r in roots):
        return 1e100
    if len(set(round(r, 9) for r in roots)) != 12:
        return 1e100
    poly = _build_poly(roots, m)
    if not np.all(np.isfinite(poly)):
        return 1e100
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e100
    return -math.log(norm_sq)


def construct():
    m = 12

    # Seed roots: geometric, ratio 2 (the original construction).
    seed_roots = _geo_roots(4.0, 2.0)
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0
    best_roots = seed_roots[:]
    best_val = seed_val

    # Multistart over log(r0), log(q). Includes the seed point (4, 2).
    r0_starts = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    q_starts = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0]

    starts = []
    for r0 in r0_starts:
        for q in q_starts:
            starts.append((math.log(r0), math.log(q)))

    # Deterministic extra starts (small jitter) for robustness.
    rng = np.random.default_rng(12345)
    for _ in range(24):
        lr0 = float(rng.uniform(math.log(0.2), math.log(64.0)))
        lq = float(rng.uniform(math.log(0.4), math.log(8.0)))
        starts.append((lr0, lq))

    for (lr0, lq) in starts:
        if _time_left() < 45:
            break
        x0 = np.array([lr0, lq])
        try:
            res = minimize(
                _objective, x0,
                method='L-BFGS-B',
                bounds=[(math.log(0.05), math.log(200.0)),
                        (math.log(0.25), math.log(8.0))],
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10},
            )
        except Exception:
            continue
        try:
            cand_roots = _geo_roots(math.exp(res.x[0]), math.exp(res.x[1]))
        except Exception:
            continue
        if len(cand_roots) != 12 or any(r <= 0 for r in cand_roots):
            continue
        if len(set(round(r, 9) for r in cand_roots)) != 12:
            continue
        if any(r > 1e15 for r in cand_roots):
            continue
        cv = _c4_bound(cand_roots, m)
        if np.isfinite(cv) and cv < best_val:
            best_val = cv
            best_roots = cand_roots[:]

    # Final high-precision polish from the best point.
    if _time_left() > 30 and all(r > 0 for r in best_roots):
        r0b = best_roots[0]
        qb = best_roots[1] / best_roots[0]
        if r0b > 0 and qb > 0:
            x0 = np.array([math.log(r0b), math.log(qb)])
            try:
                res = minimize(
                    _objective, x0,
                    method='L-BFGS-B',
                    bounds=[(math.log(0.05), math.log(200.0)),
                            (math.log(0.25), math.log(8.0))],
                    options={'maxiter': 6000, 'ftol': 1e-16, 'gtol': 1e-12},
                )
                cand_roots = _geo_roots(math.exp(res.x[0]), math.exp(res.x[1]))
                if (len(cand_roots) == 12 and all(r > 0 for r in cand_roots)
                        and len(set(round(r, 9) for r in cand_roots)) == 12
                        and all(r <= 1e15 for r in cand_roots)):
                    cv = _c4_bound(cand_roots, m)
                    if np.isfinite(cv) and cv < best_val:
                        best_val = cv
                        best_roots = cand_roots[:]
            except Exception:
                pass

    # Safety: fall back to the seed if anything went wrong.
    if len(best_roots) != 12 or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
    if len(set(round(r, 9) for r in best_roots)) != 12:
        best_roots = seed_roots[:]

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
