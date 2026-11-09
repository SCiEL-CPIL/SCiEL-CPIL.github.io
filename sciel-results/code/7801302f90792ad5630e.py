"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize_scalar, minimize

_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1, num_quad=2000):
    """
    Compute the C4 uncertainty upper bound for given double-root locations.
    C4 = (∫₀^∞ |P(x)|² x^(2n-1) e^(-x) dx / Gamma(2n))^(1/2)
    where P(x) = ∏ᵢ (x - rᵢ)²
    """
    try:
        x_quad, w_quad = laggauss(num_quad)
        
        alpha = 2 * n - 1
        
        # Compute |P(x)|² = ∏ᵢ (x - rᵢ)⁴ in log space
        log_abs_P_sq = np.zeros_like(x_quad)
        for r in roots:
            diff = np.abs(x_quad - r)
            log_abs_P_sq += 4.0 * np.log(diff + 1e-300)
        
        weight_factor = np.power(x_quad, alpha)
        
        integrand = np.exp(log_abs_P_sq) * weight_factor
        integral_P = np.sum(w_quad * integrand)
        
        log_norm = gammaln(2 * n)
        norm = np.exp(log_norm)
        
        c4_bound = np.sqrt(integral_P / norm)
        
        return c4_bound
    except Exception:
        return np.inf

def _objective(roots_flat, m=12, n=1):
    """Objective function for optimization: the C4 bound."""
    roots = np.asarray(roots_flat)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def _get_laguerre_nodes(m, alpha):
    """Get generalized Laguerre-Gauss nodes with weight x^alpha * e^(-x)."""
    try:
        x, w = laggauss(m)
        # For alpha != 0, we need to adjust. Use scipy's approach:
        # The nodes of L_m^(alpha) are not simply related to standard Laguerre nodes.
        # Instead, use the roots of the generalized Laguerre polynomial directly.
        L = genlaguerre(m, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        return roots
    except Exception:
        # Fallback: use standard Laguerre nodes
        x, w = laggauss(m)
        return np.sort(x)

def _objective_alpha(alpha, m=12, n=1):
    """Objective as function of alpha: get nodes from L_m^(alpha) and compute C4 bound."""
    try:
        roots = _get_laguerre_nodes(m, alpha)
        if len(roots) < m:
            return 1e10
        roots = roots[:m]
        if np.any(roots <= 0):
            return 1e10
        return _compute_c4_bound(roots, m, n, num_quad=1000)
    except Exception:
        return 1e10

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Strategy: 
    1. Start with roots of L_{2m}^{(2n-1)}(x) (the canonical construction)
    2. Optimize the alpha parameter in a generalized Laguerre node representation
    3. Refine with direct root optimization
    """
    m = 12
    n = 1
    
    # Step 1: Initial roots from L_{2m}^{(2n-1)}
    degree = 2 * m
    alpha_init = 2 * n - 1
    L = genlaguerre(degree, alpha_init)
    roots = L.roots()
    roots = roots[np.real(roots) > 0]
    roots = np.real(roots)
    roots = np.sort(roots)
    if len(roots) >= m:
        best_roots = roots[:m].copy()
    else:
        best_roots = np.array([4.0 * (i + 1) for i in range(m)], dtype=float)
    
    best_score = _compute_c4_bound(best_roots, m, n, num_quad=1000)
    
    # Step 2: Optimize alpha for generalized Laguerre nodes
    if not _deadline_reached():
        try:
            # Search over alpha in a reasonable range
            alpha_result = minimize_scalar(
                _objective_alpha, 
                bounds=(0.5, 8.0), 
                args=(m, n),
                method='bounded',
                options={'maxiter': 50, 'xatol': 0.01}
            )
            if alpha_result.fun < best_score:
                best_roots = _get_laguerre_nodes(m, alpha_result.x)
                best_score = alpha_result.fun
        except Exception:
            pass
    
    # Step 3: Direct root optimization using BFGS
    if not _deadline_reached():
        try:
            def _obj_for_minimize(x):
                return _objective(x, m, n)
            
            # Use multiple starting points
            start_points = [best_roots.copy()]
            
            # Add perturbed starting points
            for scale in [0.5, 1.5, 2.0]:
                perturbed = best_roots * scale
                start_points.append(perturbed)
            
            # Log-spaced starting points
            log_roots = np.logspace(np.log10(0.1), np.log10(50), m)
            start_points.append(log_roots)
            
            best_opt_roots = best_roots.copy()
            best_opt_score = best_score
            
            for sp in start_points:
                if _deadline_reached():
                    break
                try:
                    result = minimize(
                        _obj_for_minimize,
                        sp,
                        method='L-BFGS-B',
                        bounds=[(1e-6, None)] * m,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if result.fun < best_opt_score:
                        best_opt_score = result.fun
                        best_opt_roots = result.x.copy()
                except Exception:
                    continue
            
            if best_opt_score < best_score:
                best_roots = best_opt_roots
                best_score = best_opt_score
        except Exception:
            pass
    
    # Step 4: Final refinement with Nelder-Mead on best solution
    if not _deadline_reached():
        try:
            result = minimize(
                lambda x: _objective(x, m, n),
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_score:
                best_roots = result.x.copy()
                best_score = result.fun
        except Exception:
            pass
    
    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

def _fallback_construct(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)
