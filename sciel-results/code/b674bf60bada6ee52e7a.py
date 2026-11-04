"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Laguerre quadrature nodes and weights
# We need enough nodes to accurately integrate the product of 12 squared quadratics (degree 48)
# Degree 48 requires at least 25 nodes for exact integration, but since roots are variable,
# we need a fixed set of nodes that covers the support well.
# The support is roughly [0, ~50]. 
# Let's use 150 nodes for high accuracy.
_GL_N = 150
_GL_X, _GL_W = roots_laguerre(_GL_N)
# Weight function for n=1 is x * exp(-x)
# In Gauss-Laguerre, the weight is exp(-x). So we multiply by x explicitly.
_GL_WX = _GL_X * _GL_W

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    roots: array of 12 positive floats.
    """
    roots = np.asarray(roots, dtype=float)
    # P(x) = prod (x - r_i)^2
    # P(x)^2 = prod (x - r_i)^4
    # We compute log(P^2) to avoid overflow/underflow
    # log(P^2) = 4 * sum log|x - r_i|
    
    # Vectorized computation
    # xq shape (N,), roots shape (M,)
    # diff shape (N, M)
    diff = _GL_X[:, np.newaxis] - roots[np.newaxis, :]
    
    # Avoid log(0)
    abs_diff = np.abs(diff)
    # Add a small epsilon to prevent log(0) if a root coincides with a node
    # This is unlikely with random roots, but good for safety
    abs_diff_safe = np.maximum(abs_diff, 1e-15)
    
    log_p_sq = 4.0 * np.sum(np.log(abs_diff_safe), axis=1)
    
    p_sq = np.exp(log_p_sq)
    
    # Integrate P(x)^2 * x * exp(-x)
    # The quadrature weights _GL_W already include exp(-x)
    # So we need to multiply by x (which is in _GL_X)
    integrand = p_sq * _GL_WX
    
    int0 = np.sum(integrand)
    int4 = np.sum(integrand * _GL_X**4)
    
    if int0 < 1e-30:
        return 1e30
        
    return int4 / (int0 * int0)

def _objective_log(y):
    """
    Objective function in log-space.
    y: array of 12 log-roots.
    """
    # Map back to roots
    roots = np.exp(y)
    return _compute_c4_bound(roots)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    best_roots = None
    best_score = 1e30
    
    # Strategy: Optimize in log-space using L-BFGS-B
    # Initial guesses for log-roots
    # The roots are expected to be in the range [1, 30] roughly.
    # Log range: [0, 3.4]
    
    initializations = []
    
    # 1. Uniform in log-space
    for _ in range(10):
        initializations.append(np.random.uniform(0.0, 3.5, 12))
        
    # 2. Based on Laguerre roots (log-scaled)
    lag_zeros, _ = roots_laguerre(12)
    # Laguerre roots for n=1 are shifted.
    # Typical scale is ~ m.
    initializations.append(np.log(lag_zeros + 1.0))
    
    # 3. Clustered guesses
    np.random.seed(42)
    for center_log in [0.5, 1.0, 1.5, 2.0]:
        for spread_log in [0.5, 1.0, 1.5]:
            initializations.append(center_log + spread_log * np.random.randn(12))
            
    # 4. Random restarts
    for _ in range(20):
        initializations.append(np.random.uniform(-1.0, 4.0, 12))

    # Evaluate and refine
    for init_y in initializations:
        if time.time() > deadline - 60:
            break
            
        try:
            # Local refinement using L-BFGS-B
            # L-BFGS-B is efficient for this smooth problem
            res = minimize(
                _objective_log,
                init_y,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                # Convert back to roots
                best_roots = np.exp(res.x)
        except Exception:
            continue

    # If L-BFGS-B failed or didn't find anything, try Nelder-Mead on log-space
    if best_roots is None or best_score > 1e20:
        for init_y in initializations[:10]:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    _objective_log,
                    init_y,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_roots = np.exp(res.x)
            except Exception:
                pass

    # Final check and cleanup
    if best_roots is None:
        # Fallback to simple heuristic
        return [4.0 * (i + 1) for i in range(12)]
        
    # Ensure roots are positive
    result = [max(float(r), 1e-6) for r in best_roots]
    
    if len(result) != 12:
        result = [4.0 * (i + 1) for i in range(12)]
        
    return result
# EVOLVE-BLOCK-END
