"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import differential_evolution, minimize
import time

def _compute_c4_stable(roots):
    """Compute C4 upper bound using log-sum-exp for numerical stability."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    n_quad = 250
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log|P(x)| = 2 * sum_i log|x_i - r_i|
    # Vectorized: (n_quad, 12)
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs, axis=1)
    
    # int_P = sum w_i * exp(log_P_i)
    # Use log-sum-exp for stability
    max_log_P = np.max(log_P)
    if np.isinf(max_log_P):
        return np.inf
    log_int_P = max_log_P + np.log(np.sum(w_quad * np.exp(log_P - max_log_P)))
    
    # int_P2 = sum w_i * exp(2*log_P_i)
    max_2log_P = np.max(2.0 * log_P)
    if np.isinf(max_2log_P):
        return np.inf
    log_int_P2 = max_2log_P + np.log(np.sum(w_quad * np.exp(2.0 * log_P - max_2log_P)))
    
    c4 = np.exp(log_int_P2 - 2.0 * log_int_P)
    return c4

def _objective_direct(x):
    """Objective for direct 12D optimization in log-space."""
    roots = np.exp(x)
    if np.any(roots > 1e8) or np.any(roots < 1e-8):
        return 1e10
    return _compute_c4_stable(roots)

def _objective_geometric(params):
    """Objective for geometric parameterization: roots = r0 * q^i."""
    log_r0, log_q = params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    roots = r0 * (q ** np.arange(12))
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return 1e10
    return _compute_c4_stable(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 590
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_score = _compute_c4_stable(best_roots)
    
    if time.time() > deadline - 30:
        return [float(xi) for xi in best_roots]
    
    # === Phase 1: Geometric parameterization with Differential Evolution ===
    # roots_i = r0 * q^i, optimize (log_r0, log_q)
    
    # Fit initial geometric parameters from Laguerre roots
    log_roots0 = np.log(x0)
    indices = np.arange(12)
    coeffs = np.polyfit(indices, log_roots0, 1)
    log_q_init = coeffs[0]
    log_r0_init = coeffs[1]
    
    # DE bounds for geometric: log_r0 in [-5, 5], log_q in [-1, 2]
    geo_bounds = [(-5, 5), (-1, 2)]
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective_geometric,
                bounds=geo_bounds,
                maxiter=200,
                popsize=15,
                tol=1e-12,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                r0 = np.exp(de_result.x[0])
                q = np.exp(de_result.x[1])
                candidate = r0 * (q ** np.arange(12))
                if np.all(candidate > 0) and np.all(candidate < 1e6):
                    candidate = np.sort(candidate)
                    score = _compute_c4_stable(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # === Phase 2: Full 12D Differential Evolution in log-space ===
    if time.time() < deadline - 30:
        # Bounds for log-space: roots in [1e-4, 1e4]
        log_bounds = [(-9, 9)] * 12
        
        # Initial population: mix of Laguerre, geometric, and random
        x0_log = np.log(best_roots)
        
        try:
            de_result = differential_evolution(
                _objective_direct,
                bounds=log_bounds,
                maxiter=100,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol',
                x0=x0_log
            )
            if de_result.fun < best_score:
                candidate = np.exp(de_result.x)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_stable(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # === Phase 3: Powell refinement (derivative-free) ===
    if time.time() < deadline - 10:
        try:
            log_roots = np.log(best_roots)
            result = minimize(
                _objective_direct,
                log_roots,
                method='Powell',
                bounds=[(-9, 9)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'xtol': 1e-14}
            )
            candidate = np.exp(result.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_stable(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # === Phase 4: Nelder-Mead final polish ===
    if time.time() < deadline - 5:
        try:
            log_roots = np.log(best_roots)
            result = minimize(
                _objective_direct,
                log_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            candidate = np.exp(result.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_stable(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure all roots are positive and finite
    best_roots = np.asarray(best_roots, dtype=np.float64)
    best_roots = np.maximum(best_roots, 1e-10)
    best_roots = np.minimum(best_roots, 1e6)
    best_roots = np.sort(best_roots)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
