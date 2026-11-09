"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre, gammaln, roots_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1, num_quad=200):
    """
    Compute the C4 uncertainty upper bound for given double-root locations.
    C4 = (∫₀^∞ |P(x)|² x^(2n-1) e^(-x) dx)^(1/2) / (∫₀^∞ x^(2n-1) e^(-x) dx)^(1/2)
    where P(x) = ∏ᵢ (x - rᵢ)²
    """
    try:
        x_quad, w_quad = np.polynomial.laguerre.laggauss(num_quad)
        
        log_abs_P_sq = np.zeros_like(x_quad)
        for r in roots:
            log_abs_P_sq += 4.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
        alpha = 2 * n - 1
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

def _get_initial_roots_squared_laguerre(m=12, n=1):
    """
    Get initial root positions from the zeros of (L_m^(alpha)(x))^2.
    The zeros of (L_m^(alpha))^2 are the same as the zeros of L_m^(alpha),
    but this representation emphasizes the double-root structure.
    For m=12, n=1, alpha = 2n-1 = 1.
    """
    try:
        alpha = 2 * n - 1  # = 1
        # Get zeros of L_m^(alpha)(x) - these become the double roots
        roots, _ = roots_genlaguerre(m, alpha)
        roots = np.sort(roots)
        # Filter to positive real roots
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        if len(roots) == m:
            return roots
        else:
            # Fallback to higher degree
            return _get_initial_roots_higher_degree(m, n)
    except Exception:
        return _get_initial_roots_higher_degree(m, n)

def _get_initial_roots_higher_degree(m=12, n=1):
    """Get initial root positions using the higher-degree Laguerre polynomial construction."""
    try:
        degree = 2 * m
        alpha = 2 * n - 1
        L = genlaguerre(degree, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        if len(roots) >= m:
            return roots[:m]
        else:
            while len(roots) < m:
                roots = np.append(roots, roots[-1] + 1.0)
            return roots
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

def _optimize_roots(roots_init, m=12, n=1, method='trust-constr', maxiter=500):
    """
    Optimize root positions to minimize the C4 bound.
    Uses trust-constr for constrained optimization.
    """
    try:
        bounds = [(1e-8, None)] * m
        
        if method == 'trust-constr':
            # Use trust-constr with bounds
            result = minimize(
                _objective,
                roots_init,
                args=(m, n),
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            return result.x, result.fun
        elif method == 'L-BFGS-B':
            result = minimize(
                _objective,
                roots_init,
                args=(m, n),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x, result.fun
        elif method == 'Nelder-Mead':
            result = minimize(
                _objective,
                roots_init,
                args=(m, n),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            return result.x, result.fun
        else:
            return roots_init, _objective(roots_init, m, n)
    except Exception:
        return roots_init, _objective(roots_init, m, n)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Uses the zeros of (L_12^(1)(x))^2 as initial positions (emphasizing double-root structure),
    then refines with a constrained trust-region optimizer to minimize the C4 moment ratio.
    """
    m = 12
    n = 1
    
    # Get initial roots from squared Laguerre polynomial zeros
    roots_init = _get_initial_roots_squared_laguerre(m, n)
    roots_init = np.maximum(roots_init, 1e-6)
    roots_init = np.sort(roots_init)
    
    best_roots = roots_init.copy()
    best_score = _objective(roots_init, m, n)
    
    # Try multiple optimization strategies
    strategies = [
        ("trust-constr", 300),
        ("L-BFGS-B", 300),
        ("Nelder-Mead", 200),
    ]
    
    for method_name, maxiter in strategies:
        if _deadline_reached():
            break
        
        try:
            opt_roots, opt_score = _optimize_roots(roots_init, m, n, method=method_name, maxiter=maxiter)
            
            if opt_score < best_score:
                best_roots = opt_roots
                best_score = opt_score
                # Update initial for next round
                roots_init = opt_roots.copy()
        except Exception:
            continue
    
    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

def _fallback_construct(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m, alpha=2*n-1)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

# EVOLVE-BLOCK-END
