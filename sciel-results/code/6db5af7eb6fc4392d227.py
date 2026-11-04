"""Refined Cohn-Goncalves Laguerre construction: 12-dim log-space DE + L-BFGS."""
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


def _objective_logspace(y):
    """
    Objective function for 12-dim log-space optimization.
    y: array of 12 log-roots.
    Returns C4 bound.
    """
    if not np.all(np.isfinite(y)):
        return 1e10
    roots = np.exp(y)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    return _c4_bound(roots)


def construct():
    m = 12
    n_roots = 12
    
    # Seed roots: geometric spacing
    seed_roots = np.array([4.0 * (i + 1) for i in range(n_roots)])
    seed_logs = np.log(seed_roots)
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Bounds for log-roots
    # Roots likely in [0.1, 1000]
    log_lower = np.full(n_roots, math.log(0.1))
    log_upper = np.full(n_roots, math.log(1000.0))
    bounds = list(zip(log_lower, log_upper))
    
    # Phase 1: Differential Evolution in 12-dim log-space
    # DE is expensive in 12D, so we limit iterations and population
    # Time budget: ~300s
    if _time_left() > 300:
        try:
            # Use Sobol sequence for better initial distribution
            de_result = differential_evolution(
                _objective_logspace,
                bounds=bounds,
                seed=42,
                maxiter=50,  # Reduced maxiter due to 12D cost
                tol=1e-8,
                popsize=20,  # Larger population for 12D
                polish=False,  # We do polish manually to control time
                init='sobol',
                updating='deferred',
                workers=1,
                maxfevs=5000, # Cap function evaluations
            )
            
            if de_result.fun < best_val:
                best_logs = de_result.x
                best_roots = np.exp(best_logs)
                best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B in 12-dim log-space
    # Start from the best found so far (DE result or seed)
    if _time_left() > 60:
        x0 = np.log(best_roots) if not np.all(np.isfinite(best_roots)) or np.any(best_roots <= 0) else np.log(best_roots)
        # Ensure x0 is within bounds
        x0 = np.clip(x0, log_lower, log_upper)
        
        try:
            res = minimize(
                _objective_logspace,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_roots = np.exp(res.x)
        except Exception:
            pass
    
    # Phase 3: Try a second polish from a slightly perturbed point to escape local minima
    if _time_left() > 30:
        # Perturb the best solution slightly
        noise = np.random.normal(0, 0.1, size=n_roots)
        x0_perturbed = np.clip(np.log(best_roots) + noise, log_lower, log_upper)
        
        try:
            res2 = minimize(
                _objective_logspace,
                x0=x0_perturbed,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if res2.fun < best_val and np.all(np.isfinite(res2.x)):
                best_val = res2.fun
                best_roots = np.exp(res2.x)
        except Exception:
            pass
    
    # Final check
    if not np.all(np.isfinite(best_roots)) or np.any(best_roots <= 0):
        return [4.0 * (i + 1) for i in range(n_roots)]
        
    return [float(r) for r in best_roots]
