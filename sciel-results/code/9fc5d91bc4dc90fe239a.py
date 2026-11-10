# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _entropy_of_projection(P, r):
    """Shannon entropy (bits) of the projected distribution x + r*y."""
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.bincount(inv.ravel(), weights=P.ravel(), minlength=len(uniq))
    pooled = pooled[pooled > 1e-15]
    if len(pooled) == 0:
        return 0.0
    return float(-(pooled * np.log2(pooled)).sum())

def _ratio(P):
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, 0.0),
            _entropy_of_projection(P, 1.0),
            _entropy_of_projection(P, 2.0))
    if M <= 1e-9:
        return -np.inf
    return D / M

def _objective(params):
    """
    Objective function for minimization.
    params: 100-element array of log-weights.
    """
    w = params
    w_shifted = w - np.max(w)
    p = np.exp(w_shifted)
    s = p.sum()
    if s < 1e-300:
        return 1e10
    P = p / s
    P = P.reshape(10, 10)
    r = _ratio(P)
    if not np.isfinite(r):
        return 1e10
    return -r

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    bounds = [(-20, 20) for _ in range(n * n)]
    
    best_P = None
    best_score = -np.inf
    
    # Run Differential Evolution multiple times with different seeds
    seeds = [1, 2, 3]
    
    for seed in seeds:
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=seed,
                maxiter=150,
                popsize=20,
                tol=1e-6,
                init='sobol'
            )
            
            # Check if result is valid
            if res.fun < 1e9:
                log_w_opt = res.x
                w = log_w_opt - np.max(log_w_opt)
                p = np.exp(w)
                P_opt = p / p.sum()
                P_opt = P_opt.reshape(10, 10)
                score = _ratio(P_opt)
                
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_P = P_opt
        except Exception:
            continue
            
    # Fallback: Random Search if DE failed
    if best_P is None:
        best_score = -np.inf
        for _ in range(1000):
            w = np.random.uniform(-10, 10, (n, n))
            w = w - np.max(w)
            p = np.exp(w)
            P = p / p.sum()
            score = _ratio(P)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_P = P

    # Final safety checks
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
        
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
        
    return P
# EVOLVE-BLOCK-END
