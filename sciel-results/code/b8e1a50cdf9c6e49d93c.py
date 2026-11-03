"""Refined Cohn-Goncalves Laguerre construction: low-dim concave-root ansatz + DE + L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _roots_from_params(params):
    """
    Generate 12 positive double-root locations from 3 parameters:
    log_a: log of first root
    log_b: log of last root  
    c: curvature exponent (roots = a * (b/a)^(i^c / 11^c) for i=0..11)
    
    c=1 gives geometric spacing, c<1 gives concave (spaced more at start), c>1 gives convex.
    """
    log_a, log_b, c = params
    a = math.exp(log_a)
    b = math.exp(log_b)
    
    if a <= 0 or b <= 0 or b <= a:
        return None
    
    n = 12
    idx = np.arange(n, dtype=float)
    # Normalize index to [0, 1]
    t = idx / (n - 1)
    # Apply curvature
    t_c = t ** c
    # Interpolate in log-space with curvature
    log_roots = log_a + (log_b - log_a) * t_c
    roots = np.exp(log_roots)
    
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return None
    return roots


def _objective_params(params):
    """Objective: C4 bound from 3-parameter root ansatz."""
    roots = _roots_from_params(params)
    if roots is None:
        return 1e10
    return _c4_bound(roots)


def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Differential Evolution over 3-parameter space
    if _time_left() > 180:
        bounds = [
            (math.log(0.1), math.log(50.0)),    # log_a
            (math.log(1.0), math.log(200.0)),   # log_b (must be > log_a)
            (0.3, 3.0),                          # c (curvature)
        ]
        
        def _de_obj(params):
            roots = _roots_from_params(params)
            if roots is None:
                return 1e10
            return _c4_bound(roots)
        
        try:
            de_result = differential_evolution(
                _de_obj,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=15,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1,
            )
            
            if de_result.fun < best_val:
                roots = _roots_from_params(de_result.x)
                if roots is not None and np.all(roots > 0):
                    score = _c4_bound(roots, m)
                    if score < best_val:
                        best_val = score
                        best_roots = [float(r) for r in roots]
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B on the 3-parameter space
    if _time_left() > 60:
        # Try multiple starting points near the best found
        start_points = []
        
        # Fit to seed roots
        seed_logs = np.log(seed_roots)
        x_idx = np.arange(n_roots, dtype=float)
        t = x_idx / (n_roots - 1)
        
        # Try c=1 (geometric)
        A = np.vstack([np.ones_like(t), t]).T
        try:
            log_a_fit, log_b_fit = np.linalg.lstsq(A, seed_logs, rcond=None)[0]
            start_points.append([log_a_fit, log_b_fit, 1.0])
        except Exception:
            pass
        
        # Try various curvature values
        for c_val in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            t_c = t ** c_val
            A = np.vstack([np.ones_like(t_c), t_c]).T
            try:
                log_a_f, log_b_f = np.linalg.lstsq(A, seed_logs, rcond=None)[0]
                start_points.append([log_a_f, log_b_f, c_val])
            except Exception:
                pass
        
        # Add the DE result if available
        if best_roots != seed_roots:
            best_logs = np.log(best_roots)
            for c_val in [0.5, 0.7, 1.0, 1.3, 1.5]:
                t_c = t ** c_val
                A = np.vstack([np.ones_like(t_c), t_c]).T
                try:
                    log_a_f, log_b_f = np.linalg.lstsq(A, best_logs, rcond=None)[0]
                    start_points.append([log_a_f, log_b_f, c_val])
                except Exception:
                    pass
        
        for sp in start_points:
            if _time_left() < 30:
                break
            try:
                res = minimize(
                    _objective_params,
                    x0=np.array(sp, dtype=float),
                    method='L-BFGS-B',
                    bounds=[(math.log(0.05), math.log(100.0)),
                            (math.log(0.5), math.log(300.0)),
                            (0.2, 4.0)],
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                if res.fun < best_val:
                    roots = _roots_from_params(res.x)
                    if roots is not None and np.all(roots > 0):
                        score = _c4_bound(roots, m)
                        if score < best_val:
                            best_val = score
                            best_roots = [float(r) for r in roots]
            except Exception:
                continue
    
    # Phase 3: Direct 12-dim local refinement from best
    if _time_left() > 40:
        def obj12(roots_arr):
            if np.any(roots_arr <= 0):
                return 1e10
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_low = np.ones(n_roots) * 0.01
        bounds_high = np.ones(n_roots) * 500.0
        
        try:
            res = minimize(
                obj12,
                x0=x0,
                method='L-BFGS-B',
                bounds=list(zip(bounds_low, bounds_high)),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.fun < best_val and np.all(res.x > 0):
                best_val = res.fun
                best_roots = [float(r) for r in res.x]
        except Exception:
            pass
    
    return best_roots
