"""Refined Cohn-Goncalves Laguerre sign-uncertainty root optimizer."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import laguerre

_M = 12
_N = 1
_DEADLINE_S = 630.0
_SAFETY_S = 60.0


def _build_polynomial(roots):
    """
    Build the Laguerre combination polynomial with double roots at the given locations.
    The refined construction: P(x) = product_i ( (d/dx L_{2i}^{(1)}(x))^2 weighted ) ...
    We use the standard double-root form: each root r_i contributes a factor
    ( L_{i}^{(1)}(x) - L_{i}^{(1)}(r_i) )^2 scaled, ensuring a double root at r_i.
    """
    x = np.linspace(1e-6, max(roots) * 1.5 + 50.0, 4000)
    logP = np.zeros_like(x)
    for i, r in enumerate(roots):
        # L_{i+1}^{(1)} evaluated; subtract its value at r to force a root at r.
        li = laguerre(i + 1, 1)
        f = li(x) - li(r)
        # double root -> square
        logP += 2.0 * np.log(np.abs(f) + 1e-300)
    return x, logP


def _sign_changes(x, logP):
    s = np.sign(np.exp(logP - np.max(logP)))
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def _bound_from_roots(roots):
    """
    Upper bound on C4. The refined bound: C4_upper = (m+1) / x_last,
    where x_last is the location of the last sign change of the constructed
    polynomial (larger x_last => smaller bound => better). We maximize x_last.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return np.inf
    x, logP = _build_polynomial(roots)
    s = np.sign(np.exp(logP - np.max(logP)))
    # find last index where sign flips
    flips = np.where(s[1:] != s[:-1])[0]
    if len(flips) == 0:
        return np.inf
    x_last = x[flips[-1] + 1]
    # Upper bound on C4:
    bound = (_M + 1.0) / x_last
    return float(bound)


def _objective(roots):
    try:
        v = _bound_from_roots(roots)
        if not np.isfinite(v):
            return 1e9
        return v
    except Exception:
        return 1e9


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE_S - _SAFETY_S

    # Seed: equally spaced (matches reference) as a safe fallback.
    best_roots = np.array([4.0 * (i + 1) for i in range(_M)])
    best_val = _bound_from_roots(best_roots)
    if not np.isfinite(best_val):
        best_val = 1e9

    def check_deadline():
        return (time.time() - t0) < deadline

    # Local refinement from the seed via Nelder-Mead on the log-spaced roots.
    def neg_obj_log(logr):
        roots = np.exp(logr)
        return _objective(roots)

    init_log = np.log(best_roots)
    try:
        res = minimize(neg_obj_log, init_log, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        if check_deadline() and res.fun < best_val:
            cand = np.exp(res.x)
            if np.all(cand > 0) and np.all(np.diff(cand) > 0):
                cv = _bound_from_roots(cand)
                if np.isfinite(cv) and cv < best_val:
                    best_val = cv
                    best_roots = cand
    except Exception:
        pass

    # Global polish with a short differential evolution if time remains.
    if check_deadline():
        try:
            # bounds on log-roots: keep them positive and ordered by sorting after.
            lo = np.log(np.array([0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 15.5, 17.0]))
            hi = np.log(np.array([2.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0]))

            def de_obj(logr):
                roots = np.sort(np.exp(logr))
                return _objective(roots)

            res = differential_evolution(de_obj, list(zip(lo, hi)), seed=0,
                                         maxiter=60, tol=1e-8, popsize=12,
                                         polish=True)
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                cv = _bound_from_roots(cand)
                if np.isfinite(cv) and cv < best_val:
                    best_val = cv
                    best_roots = cand
        except Exception:
            pass

    # Final local polish from the current best.
    if check_deadline():
        try:
            def neg_obj_log(logr):
                roots = np.sort(np.exp(logr))
                return _objective(roots)
            res = minimize(neg_obj_log, np.log(best_roots), method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-11})
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                cv = _bound_from_roots(cand)
                if np.isfinite(cv) and cv < best_val:
                    best_val = cv
                    best_roots = cand
        except Exception:
            pass

    # Ensure validity: strictly positive, strictly increasing.
    best_roots = np.sort(np.asarray(best_roots, dtype=float))
    if np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        best_roots = np.array([4.0 * (i + 1) for i in range(_M)])

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
