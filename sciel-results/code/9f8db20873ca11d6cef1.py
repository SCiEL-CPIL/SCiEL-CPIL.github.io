"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction optimizer."""
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    # Use Gauss-Laguerre quadrature with enough points
    # The integrals are over [0, infinity) with weight e^{-x}
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) at quadrature points using log-space for stability
    # P(x) = prod (x - x_i)^2
    log_P = np.zeros(n_quad)
    for r in roots:
        diff = x_quad - r
        # Handle sign carefully
        abs_diff = np.abs(diff)
        log_P += 2.0 * np.log(abs_diff + 1e-300)
    
    # Check for overflow/underflow
    max_log_P = np.max(log_P)
    if max_log_P > 700:
        return np.inf
    
    P = np.exp(log_P)
    P2 = P * P
    
    # Integrate with weight e^{-x} using Gauss-Laguerre (which includes e^{-x})
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2) or np.isnan(int_P) or np.isnan(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    if np.isnan(c4) or np.isinf(c4) or c4 < 0:
        return np.inf
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound(roots)

def _objective_log(log_roots):
    """Objective in log-space for better conditioning."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    return _compute_c4_bound(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 570  # Safety margin from 600s limit
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # Strategy 1: Local optimization from multiple starting points
    bounds = [(1e-6, 100.0)] * 12
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    initial_points = [x0.copy()]
    
    # Geometric sequence parameterization
    for ratio in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
        base = 1.0
        geo_roots = []
        for i in range(12):
            geo_roots.append(base * (ratio ** i))
        geo_roots = np.array(geo_roots)
        if np.all(geo_roots > 0) and np.max(geo_roots) < 100:
            initial_points.append(geo_roots)
    
    # Perturbed versions
    for _ in range(10):
        perturbed = x0 * (1 + 0.3 * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Log-space optimization
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
        
        # Try in log-space
        try:
            log_init = np.log(x_init)
            result_log = minimize(
                _objective_log,
                log_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result_log.fun < best_score:
                candidate = np.exp(result_log.x)
                candidate = np.sort(candidate)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
        
        # Try in linear space
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                candidate = result.x
                candidate = np.sort(candidate)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 30:
        return [float(xi) for xi in best_roots]
    
    # Strategy 2: Differential Evolution for global exploration
    try:
        de_bounds = [(1e-3, 50.0)] * 12
        result_de = differential_evolution(
            _objective,
            bounds=de_bounds,
            seed=42,
            maxiter=100,
            tol=1e-10,
            polish=False,
            workers=1
        )
        if result_de.fun < best_score:
            candidate = np.sort(result_de.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                score = _compute_c4_bound(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return [float(xi) for xi in best_roots]
    
    # Final polish with tighter tolerances
    try:
        result_final = minimize(
            _objective,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if result_final.fun < best_score:
            candidate = np.sort(result_final.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                score = _compute_c4_bound(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    except Exception:
        pass
    
    return [float(xi) for xi in best_roots]
