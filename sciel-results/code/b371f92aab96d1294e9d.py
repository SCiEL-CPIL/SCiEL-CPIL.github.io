"""Refined Cohn-Goncalves Laguerre construction: 12-DOF log-root optimization with multi-restart L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize

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


def _objective_logroots(log_roots, m=12):
    """Objective in log-root space: C4 bound."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    return _c4_bound(roots, m)


def construct():
    m = 12
    n_roots = 12
    
    # Seed: geometric roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_log_roots = np.log(seed_roots)
    seed_val = _c4_bound(seed_roots, m)
    
    best_val = seed_val
    best_log_roots = seed_log_roots.copy()
    
    # Phase 1: Multi-restart L-BFGS in log-root space (12 DOF)
    # Generate diverse starting points
    start_points = [seed_log_roots.copy()]
    
    # Log-normal distributions with various parameters
    z_quantiles = np.array([(-2.5, -1.8, -1.3, -0.9, -0.5, -0.1, 0.3, 0.7, 1.1, 1.5, 2.0, 2.8)])
    for mu in np.linspace(-1.0, 3.0, 7):
        for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
            log_roots = mu + sigma * z_quantiles[0]
            if np.all(log_roots > -20) and np.all(log_roots < 20):
                start_points.append(log_roots)
    
    # Arithmetic-like spacings in log space
    for log_base in np.linspace(0.3, 2.0, 6):
        for log_offset in np.linspace(-1.0, 2.0, 5):
            log_roots = log_offset + log_base * np.arange(n_roots)
            if np.all(log_roots > -20) and np.all(log_roots < 20):
                start_points.append(log_roots)
    
    # Perturbations of seed
    rng = np.random.RandomState(42)
    for _ in range(20):
        perturbation = rng.randn(n_roots) * 0.5
        log_roots = seed_log_roots + perturbation
        if np.all(log_roots > -20) and np.all(log_roots < 20):
            start_points.append(log_roots)
    
    bounds_log = [(-20.0, 20.0)] * n_roots
    
    for i, x0 in enumerate(start_points):
        if _time_left() < 30:
            break
        try:
            res = minimize(
                _objective_logroots,
                x0=np.array(x0, dtype=float),
                args=(m,),
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_log_roots = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish with tighter tolerances from the best point
    if _time_left() > 60:
        for tol in [1e-16, 1e-17]:
            if _time_left() < 30:
                break
            try:
                res = minimize(
                    _objective_logroots,
                    x0=best_log_roots.copy(),
                    args=(m,),
                    method='L-BFGS-B',
                    bounds=bounds_log,
                    options={'maxiter': 1000, 'ftol': tol, 'gtol': 1e-14, 'maxls': 100},
                )
                if res.fun < best_val and np.all(np.isfinite(res.x)):
                    best_val = res.fun
                    best_log_roots = res.x.copy()
            except Exception:
                continue
    
    # Phase 3: Nelder-Mead polish (no gradient, can escape shallow local minima)
    if _time_left() > 40:
        try:
            res = minimize(
                _objective_logroots,
                x0=best_log_roots.copy(),
                args=(m,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_log_roots = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS polish from Nelder-Mead result
    if _time_left() > 30:
        try:
            res = minimize(
                _objective_logroots,
                x0=best_log_roots.copy(),
                args=(m,),
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 200},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_log_roots = res.x.copy()
        except Exception:
            pass
    
    # Convert back to linear roots
    best_roots = np.exp(best_log_roots)
    
    if not np.all(np.isfinite(best_roots)) or np.any(best_roots <= 0):
        best_roots = np.array(seed_roots, dtype=float)
    
    return [float(r) for r in best_roots]
