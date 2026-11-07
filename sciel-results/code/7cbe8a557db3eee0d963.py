"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    # Ensure nonnegativity (should already be enforced, but just in case)
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)
    m = len(g)
    
    # Build y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2), so spacing is 1.0 / (m + 1)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Note: the sum is over i from 0 to m (inclusive), which is m+1 terms
    # y has indices 0 to m+1
    terms = (y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _negative_value(h):
    """Negative of the objective for minimization."""
    return -_evaluate(h)

# EVOLVE-BLOCK-START
def construct():
    """Construct 50 nonnegative step heights to maximize the objective."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try a few different optimization strategies
    # Strategy 1: Direct optimization with L-BFGS-B
    try:
        res = minimize(_negative_value, best_h.copy(), method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < -best_val:
            h_candidate = np.maximum(res.x, 0.0)
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return list(best_h)
    
    # Strategy 2: Try different initializations with L-BFGS-B
    np.random.seed(42)
    inits = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.ones(N) * 2.0,
        np.linspace(0.1, 1.0, N),
        np.linspace(1.0, 0.1, N),
        np.exp(-np.linspace(0, 2, N)),
        1.0 / (1.0 + np.linspace(0, 5, N)),
        np.sin(np.linspace(0, np.pi, N)) + 0.1,
        np.cos(np.linspace(0, 2*np.pi, N)) + 1.5,
        np.random.rand(N) * 2,
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        init = np.maximum(init, 0.0)
        try:
            res = minimize(_negative_value, init.copy(), method='L-BFGS-B',
                           bounds=[(0, None)] * N,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            h_candidate = np.maximum(res.x, 0.0)
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
        except Exception:
            pass
    
    if time.time() > deadline - 20:
        return list(best_h)
    
    # Strategy 3: Differential evolution for global search
    try:
        # Use a smaller population for speed
        res_de = differential_evolution(_negative_value, 
                                         bounds=[(0, 5.0)] * N,
                                         seed=42, maxiter=50, 
                                         popsize=15, tol=1e-6,
                                         polish=True,
                                         init='latinhypercube')
        if res_de.success or res_de.fun < -best_val:
            h_candidate = np.maximum(res_de.x, 0.0)
            val_candidate = _evaluate(h_candidate)
            if val_candidate > best_val:
                best_val = val_candidate
                best_h = h_candidate
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return list(best_h)
    
    # Strategy 4: Polish the best result with high-precision L-BFGS-B
    try:
        res = minimize(_negative_value, best_h.copy(), method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
        h_candidate = np.maximum(res.x, 0.0)
        val_candidate = _evaluate(h_candidate)
        if val_candidate > best_val:
            best_val = val_candidate
            best_h = h_candidate
    except Exception:
        pass
    
    # Final check
    return list(np.maximum(best_h, 0.0))
# EVOLVE-BLOCK-END
