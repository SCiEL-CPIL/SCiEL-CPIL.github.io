"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0  # wall-clock limit
_SAFETY = 30.0     # margin before the kill

def _deadline_reached():
    return (time.time() - _START) >= (_DEADLINE - _SAFETY)

def _laguerre_coeffs(poly_coeffs, order=24):
    """Expand a polynomial (highest-degree-first coeffs) in the Laguerre basis L_k(x).
    Returns the coefficient of each L_k for k=0..order."""
    # Build Vandermonde-like: express x^j in Laguerre basis.
    # L_k(x) = sum_{j=0}^k (-1)^j binom(k, j) x^j / j!  (physicists' vs mathematicians')
    # Use mathematicians' L_k via lagn; build basis matrix B[k, j] = coeff of x^j in L_k(x).
    # Then poly_coeffs (deg d down to 0) -> vector c[j] for x^j.
    # Solve B^T? Actually poly = sum_j c[j] x^j = sum_k a[k] L_k(x) = sum_k a[k] sum_j B[k,j] x^j.
    # So c[j] = sum_k a[k] B[k,j]  =>  c = B^T a  =>  a = (B^T)^{-1} c.
    B = np.zeros((order + 1, order + 1))
    for k in range(order + 1):
        # Explicit: L_k(x) = sum_{j=0}^k (-1)^j * comb(k,j) * x^j / j!
        from math import comb, factorial
        for j in range(k + 1):
            B[k, j] = (-1) ** j * comb(k, j) / factorial(j)
    # c[j] for j=0..order
    c = np.zeros(order + 1)
    # poly_coeffs highest first
    d = len(poly_coeffs) - 1
    for idx, coef in enumerate(poly_coeffs):
        power = d - idx
        if power <= order:
            c[power] = coef
    # a = solve(B.T, c)
    try:
        a = np.linalg.solve(B.T, c)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(B.T, c, rcond=None)[0]
    return a

def _poly_from_roots(roots):
    # P(x) = prod (x - r)^2, return coeffs highest first
    p = np.array([1.0])
    for r in roots:
        q = np.array([1.0, -2.0 * r, r * r])
        p = np.convolve(p, q)
    return p

def _score_roots(roots):
    # C4 bound = 4 * last sign change (largest root)
    return 4.0 * float(max(roots))

def _constraint_penalty(roots, order=24):
    p = _poly_from_roots(roots)
    a = _laguerre_coeffs(p, order)
    # penalize negative coefficients
    neg = np.maximum(0.0, -a)
    return float(np.sum(neg ** 2)) + float(np.max(np.abs(a))) * 1e-6

def _objective(x, order=24):
    # x = roots (positive). Add small floor.
    roots = np.maximum(x, 1e-6)
    base = _score_roots(roots)
    pen = _constraint_penalty(roots, order)
    return base + 100.0 * pen

def construct():
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_val = _score_roots(best_roots) + 100.0 * _constraint_penalty(best_roots)

    # Try to improve via optimization within the deadline.
    try:
        # Differential evolution is robust but slow; use a bounded search.
        bounds = [(1e-4, 60.0)] * 12
        if not _deadline_reached():
            res = differential_evolution(
                _objective, bounds, maxiter=200, tol=1e-6,
                seed=1, polish=True, init='sobol',
                update_interval=1, popsize=15,
            )
            if res.x is not None:
                cand = np.maximum(res.x, 1e-6)
                cv = _score_roots(cand) + 100.0 * _constraint_penalty(cand)
                if cv < best_val:
                    best_roots = cand
                    best_val = cv
    except Exception:
        pass

    if _deadline_reached():
        # Return best found; ensure validity (positive).
        return [float(r) for r in best_roots]

    # Local polish with Nelder-Mead on the best.
    try:
        res2 = minimize(
            _objective, best_roots, method='Nelder-Mead',
            options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10},
        )
        if res2.x is not None:
            cand = np.maximum(res2.x, 1e-6)
            cv = _score_roots(cand) + 100.0 * _constraint_penalty(cand)
            if cv < best_val:
                best_roots = cand
                best_val = cv
    except Exception:
        pass

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
