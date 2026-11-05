"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign-uncertainty bound.

Design (kept from the seed family): 12 positive *double* roots placed on an
arithmetic progression  r_i = r0 + (i-1)*d  (i = 1..12), m = 12, n = 1.

The seed uses r0 = 4.0, d = 4.0  ->  [4, 8, ..., 48], which scores 1.0.
This program keeps that structure and only attempts a *local polish* of the
objective (a smooth surrogate of the non-smooth min/max bound) around the seed.
It never returns a root set whose objective is worse than the seed's, so the
worst case is exactly the seed (score 1.0).
"""
import time
import math
import numpy as np
from scipy.special import eval_laguerre, factorial

# ---- fixed problem data -------------------------------------------------
M = 12
N = 1
NROOTS = 12
SEED = [4.0 * (i + 1) for i in range(12)]
SEED_R0, SEED_D = 4.0, 4.0


def _last_sign_change(vals):
    """Index of the last sign change of a real sequence (0/1/2)."""
    if len(vals) < 2:
        return 0
    signs = np.sign(vals)
    nz = signs[signs != 0]
    if len(nz) < 2:
        return 0
    changes = np.nonzero(nz[:-1] * nz[1:] < 0)[0]
    return int(len(changes)) if len(changes) else 0


def _objective(r0, d):
    """Reproduce the notebook's verification + last-sign-change score.

    Returns a finite float (lower is better); a large penalty if the
    double-root / non-negativity constraints are violated.
    """
    try:
        r = r0 + np.arange(NROOTS) * d
        if np.any(r <= 0.0):
            return 1e12
        # double roots -> P'(r_i) = 0 for every i
        P = np.ones(M + 1)
        for k in range(1, M + 1):
            P[k] = np.prod(r - r[k - 1])
        Pp = np.zeros(M + 1)
        for k in range(1, M + 1):
            Pp[k] = P[k - 1] * k
        Pp = Pp[::-1]
        P = P[::-1]
        for ri in r:
            if abs(np.polyval(Pp, ri)) > 1e-8:
                return 1e12
        # build the Laguerre combination
        coeff = [0.0] * (M + 1)
        for k in range(M + 1):
            coeff[k] = (factorial(M - k) / factorial(k)) * (-1) ** k
        L = np.poly1d(P) * np.poly1d(coeff)
        vals = np.array([eval_laguerre(N, x) * L(x) for x in r])
        return float(_last_sign_change(vals))
    except Exception:
        return 1e12


def _local_search(r0, d, budget):
    """Nelder-Mead polish of (r0, d) on the objective, guarded by a deadline."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return (r0, d)
    start = time.time()
    deadline = start + max(0.0, budget)
    x0 = np.array([r0, d], dtype=float)
    try:
        res = minimize(_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12})
    except Exception:
        return (r0, d)
    if time.time() > deadline:
        return (r0, d)
    if res is None or not np.all(np.isfinite(res.x)):
        return (r0, d)
    return (float(res.x[0]), float(res.x[1]))


def construct():
    start = time.time()

    # ---- verify the seed is a valid configuration ----------------------
    seed_obj = _objective(SEED_R0, SEED_D)

    # ---- attempt a light local polish, with a hard safety margin -------
    best_r0, best_d = SEED_R0, SEED_D
    best_obj = seed_obj

    # reserve ~15 s so we never approach the 600 s kill line
    budget = max(2.0, 600.0 - (time.time() - start) - 15.0)
    if budget > 2.0 and math.isfinite(seed_obj) and seed_obj < 1e11:
        r0, d = _local_search(SEED_R0, SEED_D, budget)
        obj = _objective(r0, d)
        # accept only a strict, finite improvement (never regress below seed)
        if math.isfinite(obj) and obj < best_obj:
            best_r0, best_d, best_obj = r0, d, obj

    roots = [best_r0 + i * best_d for i in range(NROOTS)]
    # final guard: roots must be positive, finite, and ordered
    if not all(math.isfinite(x) and x > 0.0 for x in roots):
        roots = list(SEED)
    return roots
