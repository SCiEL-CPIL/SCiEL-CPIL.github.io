"""Refined Cohn-Goncalves Laguerre construction: log-normal quantile roots + DE + L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

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


def _lognormal_quantiles(n=12):
    """Precompute standard normal quantiles for n equally-spaced quantiles."""
    # Use midpoints: (i+0.5)/n for i=0..n-1
    probs = (np.arange(n, dtype=float) + 0.5) / n
    return norm.ppf(probs)

# Precompute quantiles
_Z_QUANTILES = _lognormal_quantiles(12)


def _roots_from_params(params):
    """
    Generate 12 positive double-root locations from 2 parameters:
    mu: mean of log-normal (log-space center)
    sigma: std of log-normal (spread)
    
    roots_i = exp(mu + sigma * z_i) where z_i are fixed standard normal quantiles.
    This naturally produces geometric-like spacing with adjustable spread.
    """
    mu, sigma = params
    
    if sigma <= 0:
        return None
    
    log_roots = mu + sigma * _Z_QUANTILES
    roots = np.exp(log_roots)
    
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return None
    return roots


def _objective_params(params):
    """Objective: C4 bound from 2-parameter log-normal ansatz."""
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
    
    # Phase 1: Differential Evolution over 2-parameter log-normal space
    if _time_left() > 180:
        bounds = [
            (-2.0, 4.0),    # mu (log-space center)
            (0.1, 3.0),     # sigma (log-space spread)
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
                maxiter=200,
                tol=1e-10,
                popsize=20,
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
    
    # Phase 2: Local refinement with L-BFGS-B on the 2-parameter log-normal space
    if _time_left() > 60:
        # Generate diverse starting points in (mu, sigma) space
        start_points = []
        
        # Fit log-normal to seed roots
        seed_logs = np.log(seed_roots)
        mu_fit = np.mean(seed_logs)
        sigma_fit = np.std(seed_logs)
        start_points.append([mu_fit, sigma_fit])
        
        # Try a grid of (mu, sigma) values
        for mu in np.linspace(-1.0, 3.0, 8):
            for sigma in np.linspace(0.2, 2.5, 8):
                start_points.append([mu, sigma])
        
        # Add neighborhood of DE result if available
        if best_roots != seed_roots:
            best_logs = np.log(best_roots)
            mu_best = np.mean(best_logs)
            sigma_best = np.std(best_logs)
            for dm in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                for ds in [-0.2, -0.05, 0.0, 0.05, 0.2]:
                    start_points.append([mu_best + dm, max(0.05, sigma_best + ds)])
        
        for sp in start_points:
            if _time_left() < 30:
                break
            try:
                res = minimize(
                    _objective_params,
                    x0=np.array(sp, dtype=float),
                    method='L-BFGS-B',
                    bounds=[(-3.0, 5.0), (0.05, 4.0)],
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
    
    # Phase 3: Direct 12-dim local refinement from best (multiple restarts)
    if _time_left() > 40:
        def obj12(roots_arr):
            if np.any(roots_arr <= 0):
                return 1e10
            return _c4_bound(roots_arr, m)
        
        bounds_low = np.ones(n_roots) * 0.01
        bounds_high = np.ones(n_roots) * 500.0
        
        # Try from best and from perturbed versions
        x0_candidates = [np.array(best_roots, dtype=float)]
        
        # Add log-normal based candidates
        for mu in [np.mean(np.log(best_roots)) - 0.2, np.mean(np.log(best_roots)), np.mean(np.log(best_roots)) + 0.2]:
            for sigma in [0.5, 1.0, 1.5, 2.0]:
                roots_cand = np.exp(mu + sigma * _Z_QUANTILES)
                if np.all(roots_cand > 0) and np.all(roots_cand < 500):
                    x0_candidates.append(roots_cand)
        
        for x0 in x0_candidates:
            if _time_left() < 15:
                break
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
