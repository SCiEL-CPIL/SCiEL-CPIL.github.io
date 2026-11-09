"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

_START_TIME = time.time()
_TIME_LIMIT = 620.0  # safety margin before 630s kill

def _objective(r):
    """
    Objective function: approximate the C4 upper bound for the
    Cohn-Goncalves Laguerre construction with m=12, n=1.
    
    The roots r_1,...,r_12 are positive double-root locations.
    The relevant polynomial is P(x) = prod_i (x - r_i)^2.
    The C4 bound is related to the norm of P under the Laguerre weight
    w(x) = x^{n} e^{-x} = x^1 e^{-x} on [0, inf).
    
    We minimize the integral ∫_0^inf |P(x)|^2 * x * e^{-x} dx,
    which can be computed via Gauss-Laguerre quadrature.
    """
    r = np.asarray(r)
    if np.any(r <= 0) or np.any(np.isnan(r)):
        return 1e12
    
    # Use generalized Gauss-Laguerre quadrature with alpha=1 (weight x*exp(-x))
    from scipy.special import roots_genlaguerre
    try:
        nodes, weights = roots_genlaguerre(50, 1)
    except Exception:
        # Fallback: use standard and multiply by x
        from numpy.polynomial.laguerre import laggauss
        nodes, weights_std = laggauss(50)
        weights = weights_std * nodes
    
    # Compute P(x) at nodes
    log_abs_P = np.zeros_like(nodes)
    for ri in r:
        diff = nodes - ri
        log_abs_P += 2.0 * np.log(np.abs(diff) + 1e-300)
    
    val = 2.0 * log_abs_P + np.log(weights + 1e-300)
    if np.any(np.isnan(val)):
        return 1e12
    max_val = np.max(val)
    integral = np.sum(np.exp(val - max_val)) * np.exp(max_val)
    
    return integral

def _run_optimization(initial_r, max_iter=5000):
    """Run scipy optimization with time guard."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() - _START_TIME > _TIME_LIMIT:
            raise TimeoutError()
        return _objective(x)
    
    try:
        result = minimize(
            obj,
            initial_r,
            method='L-BFGS-B',
            bounds=[(0.01, None)] * 12,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x, result.fun
    except TimeoutError:
        return initial_r, _objective(initial_r)
    except Exception:
        return initial_r, _objective(initial_r)

def _multiple_restarts():
    """Run multiple random restarts to avoid local minima."""
    best_r = None
    best_val = np.inf
    
    # Starting points:
    # 1. Seed: [4, 8, ..., 48]
    # 2. Zeros of relevant Laguerre polynomial
    # 3. Random positive points
    # 4. Points near zeros of L_12(x) or related
    
    starts = []
    
    # Seed
    starts.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # Zeros of Laguerre polynomial L_12(x) (standard, alpha=0)
    from scipy.special import roots_laguerre
    try:
        zeros, _ = roots_laguerre(12)
        starts.append(zeros.copy())
    except Exception:
        pass
    
    # Zeros of generalized Laguerre L_12^(1)(x) (alpha=1)
    from scipy.special import roots_genlaguerre
    try:
        zeros_gen, _ = roots_genlaguerre(12, 1)
        starts.append(zeros_gen.copy())
    except Exception:
        pass
    
    # Random starts
    np.random.seed(42)
    for trial in range(5):
        r = np.sort(np.random.uniform(0.5, 50.0, 12))
        starts.append(r)
    
    # Spaced starts with different spacings
    for spacing in [2.0, 3.0, 5.0]:
        r = spacing * np.arange(1, 13)
        starts.append(r)
    
    # Optimized-looking: cluster some, spread others
    starts.append(np.array([1, 2, 3, 5, 8, 13, 21, 34, 40, 45, 50, 55.0]))
    
    for i, start in enumerate(starts):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        elapsed = time.time() - _START_TIME
        remaining = _TIME_LIMIT - elapsed - 30
        if remaining < 10:
            break
        
        # Scale max_iter based on remaining time
        # Rough estimate: each iteration takes ~1ms, so 10000 iter ~ 10s
        max_iter = int(min(5000, max(500, remaining * 500)))
        
        r_opt, val = _run_optimization(start, max_iter=max_iter)
        
        if val < best_val:
            best_val = val
            best_r = r_opt.copy()
    
    return best_r, best_val

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimizes the C4 upper bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    try:
        best_r, best_val = _multiple_restarts()
        
        if best_r is None:
            # Fallback to seed
            return [4.0 * (i + 1) for i in range(12)]
        
        # Ensure all are positive
        best_r = np.maximum(best_r, 0.01)
        
        # Sort for consistency
        best_r = np.sort(best_r)
        
        return [float(x) for x in best_r]
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
