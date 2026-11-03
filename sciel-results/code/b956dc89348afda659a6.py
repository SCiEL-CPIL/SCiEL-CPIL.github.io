"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    # 
    # Use Gauss-Laguerre quadrature
    # The integrals are over [0, infinity) with weight e^{-x}
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) at quadrature points
    # P(x) = prod (x - x_i)^2
    log_P = np.zeros(n_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    # Integrate with weight e^{-x} using Gauss-Laguerre (which includes e^{-x})
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 600
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # Phase 1: Global search with differential evolution
    # Use log-space parameterization to ensure positivity
    def log_objective(log_roots):
        roots = np.exp(log_roots)
        return _compute_c4_bound(roots)
    
    # Initial bounds in log-space
    log_x0 = np.log(x0)
    log_bounds = [(log_x0[i] - 2, log_x0[i] + 2) for i in range(12)]
    
    try:
        de_result = differential_evolution(
            log_objective,
            bounds=log_bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-8,
            polish=False,
            updating='immediate'
        )
        
        if de_result.fun < best_score:
            candidate = np.exp(de_result.x)
            candidate = np.sort(candidate)
            score = _compute_c4_bound(candidate)
            if score < best_score:
                best_score = score
                best_roots = candidate.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return [float(xi) for xi in best_roots]
    
    # Phase 2: Local polish with L-BFGS-B from best found so far
    bounds = [(1e-6, None)] * 12
    
    initial_points = [best_roots.copy()]
    
    # Add perturbed versions of the best found
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = best_roots * (1 + 0.1 * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    for x_init in initial_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                candidate = result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
