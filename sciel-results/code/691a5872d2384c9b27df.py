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

# Precompute Laguerre polynomial coefficients for L_1^{(1)} through L_12^{(1)}
_LAG_COEFS = []
_LAG_VALS_AT_ROOTS = []  # Will be filled per root set
for _i in range(1, _M + 1):
    _p = laguerre(_i, 1)
    _LAG_COEFS.append(_p.coef)  # highest power first


def _eval_lag_poly(coefs, x):
    """Evaluate polynomial with coefficients (highest first) at points x."""
    return np.polyval(coefs, x)


def _build_polynomial_fast(roots):
    """
    Build the Laguerre combination polynomial with double roots at the given locations.
    Uses precomputed coefficients and vectorized evaluation.
    Returns (x_grid, log_abs_P) where log_abs_P is log of |P(x)|.
    """
    max_root = max(roots)
    x_hi = max_root * 2.0 + 80.0
    x = np.linspace(1e-4, x_hi, 3000)
    
    logP = np.zeros_like(x)
    for i, r in enumerate(roots):
        coefs = _LAG_COEFS[i]
        f = _eval_lag_poly(coefs, x) - _eval_lag_poly(coefs, np.array([r]))[0]
        logP += 2.0 * np.log(np.abs(f) + 1e-300)
    return x, logP


def _find_last_sign_change(roots):
    """
    Find the x-location of the last sign change of P(x) using adaptive grid refinement.
    Returns x_last or np.inf if no sign change found.
    """
    # Coarse grid
    x_coarse, logP_coarse = _build_polynomial_fast(roots)
    P_coarse = np.sign(np.exp(logP_coarse - np.max(logP_coarse)))
    
    flips = np.where(P_coarse[1:] != P_coarse[:-1])[0]
    if len(flips) == 0:
        return np.inf
    
    last_flip_idx = flips[-1]
    x_left = x_coarse[last_flip_idx]
    x_right = x_coarse[last_flip_idx + 1]
    
    # Adaptive refinement: binary search on a fine grid in [x_left, x_right]
    x_fine = np.linspace(x_left, x_right, 500)
    # Re-evaluate on fine grid
    max_root = max(roots)
    logP_fine = np.zeros_like(x_fine)
    for i, r in enumerate(roots):
        coefs = _LAG_COEFS[i]
        f = _eval_lag_poly(coefs, x_fine) - _eval_lag_poly(coefs, np.array([r]))[0]
        logP_fine += 2.0 * np.log(np.abs(f) + 1e-300)
    
    P_fine = np.sign(np.exp(logP_fine - np.max(logP_fine)))
    flips_fine = np.where(P_fine[1:] != P_fine[:-1])[0]
    if len(flips_fine) == 0:
        # Fall back to coarse
        return x_coarse[last_flip_idx + 1]
    
    x_last = x_fine[flips_fine[-1] + 1]
    return x_last


def _bound_from_roots(roots):
    """
    Upper bound on C4. The refined bound: C4_upper = (m+1) / x_last,
    where x_last is the location of the last sign change of the constructed
    polynomial (larger x_last => smaller bound => better). We maximize x_last.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return np.inf
    x_last = _find_last_sign_change(roots)
    if not np.isfinite(x_last) or x_last <= 0:
        return np.inf
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

    def check_deadline():
        return (time.time() - t0) < deadline

    # Seed: equally spaced (matches reference) as a safe fallback.
    best_roots = np.array([4.0 * (i + 1) for i in range(_M)])
    best_val = _bound_from_roots(best_roots)
    if not np.isfinite(best_val):
        best_val = 1e9

    # Multi-start local refinement from diverse initial points
    # Generate diverse starting points in log-space
    rng = np.random.default_rng(42)
    
    # Start 1: equally spaced (current best)
    starts = [np.log(best_roots.copy())]
    
    # Start 2: log-uniform spread
    starts.append(np.linspace(np.log(0.5), np.log(30.0), _M))
    
    # Start 3: clustered at low values
    starts.append(np.log(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])))
    
    # Start 4: clustered at high values
    starts.append(np.log(np.array([5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0])))
    
    # Start 5: quadratic spacing
    starts.append(np.log(np.array([1.0 + 2.0 * (i + 1)**2 / _M for i in range(_M)])))
    
    # Start 6: random log-uniform
    starts.append(np.sort(rng.uniform(np.log(0.3), np.log(40.0), _M)))
    
    # Start 7: geometric progression
    starts.append(np.log(0.5) + np.linspace(0, np.log(60.0), _M, endpoint=False))
    
    # Start 8: mixed
    starts.append(np.log(np.array([0.8, 1.5, 2.5, 4.0, 6.0, 8.5, 11.5, 15.0, 19.0, 23.5, 28.0, 33.0])))
    
    def obj_log(logr):
        roots = np.sort(np.exp(logr))
        return _objective(roots)
    
    # Try each start with Nelder-Mead
    for idx, start_log in enumerate(starts):
        if not check_deadline():
            break
        try:
            res = minimize(obj_log, start_log, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9})
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                if np.all(cand > 0) and np.all(np.diff(cand) > 0):
                    cv = _bound_from_roots(cand)
                    if np.isfinite(cv) and cv < best_val:
                        best_val = cv
                        best_roots = cand
        except Exception:
            pass
    
    # Global polish with differential evolution if time remains
    if check_deadline():
        try:
            lo = np.log(np.array([0.3, 0.8, 1.5, 2.5, 4.0, 6.0, 8.5, 11.5, 15.0, 19.0, 23.5, 28.0]))
            hi = np.log(np.array([2.0, 4.0, 6.5, 9.5, 13.0, 17.0, 21.5, 26.0, 31.0, 36.5, 42.5, 49.0]))
            
            def de_obj(logr):
                roots = np.sort(np.exp(logr))
                return _objective(roots)
            
            res = differential_evolution(de_obj, list(zip(lo, hi)), seed=0,
                                         maxiter=40, tol=1e-8, popsize=15,
                                         polish=True, updating='deferred')
            if res.fun < best_val:
                cand = np.sort(np.exp(res.x))
                cv = _bound_from_roots(cand)
                if np.isfinite(cv) and cv < best_val:
                    best_val = cv
                    best_roots = cand
        except Exception:
            pass
    
    # Final local polish from the current best
    if check_deadline():
        try:
            res = minimize(obj_log, np.log(best_roots), method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
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
