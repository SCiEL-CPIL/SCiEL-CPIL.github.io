import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _negative_score(h):
    """Negative score for minimization."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)  # enforce nonnegativity
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return -(l2_sq / (l1 * autoconv_max))

def _find_good_support(n, k):
    """Find a good sparse support of size k in [0, n-1] that minimizes peak autoconv for uniform weights."""
    best_support = None
    best_score = -np.inf
    
    # Try several random initializations and local search
    rng = np.random.default_rng(42)
    
    for trial in range(200):
        # Start with a random set
        support = set(rng.choice(n, size=k, replace=False))
        
        # Local search: try swapping elements
        improved = True
        while improved:
            improved = False
            current_score = _score_raw(np.array([1.0 if i in support else 0.0 for i in range(n)]))
            
            for i in list(support):
                for j in range(n):
                    if j not in support:
                        new_support = set(support)
                        new_support.remove(i)
                        new_support.add(j)
                        new_score = _score_raw(np.array([1.0 if idx in new_support else 0.0 for idx in range(n)]))
                        if new_score > current_score + 1e-10:
                            support = new_support
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        final_score = _score_raw(np.array([1.0 if i in support else 0.0 for i in range(n)]))
        if final_score > best_score:
            best_score = final_score
            best_support = support
    
    return sorted(best_support) if best_support is not None else list(range(k))

def construct():
    # Step 1: Find a good sparse support via combinatorial search
    # Try different sizes of support
    best_h = None
    best_score = -np.inf
    
    for k in range(3, 12):  # Try support sizes from 3 to 11
        support = _find_good_support(N, k)
        h = np.zeros(N)
        for idx in support:
            h[idx] = 1.0 / k
        score = _score_raw(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return list(np.ones(N) / N)
    
    # Step 2: Refine with continuous optimization
    # Fix L1 = 1 by scaling, then optimize
    h0 = best_h / np.sum(best_h)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, None)] * N
    
    result = minimize(
        _negative_score,
        h0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    h_opt = np.maximum(result.x, 0.0)
    h_opt = h_opt / np.sum(h_opt)  # normalize to L1=1
    
    final_score = _score_raw(h_opt)
    
    if final_score > best_score:
        best_h = h_opt
        best_score = final_score
    
    # Return heights (the ratio is scale-invariant, so any scaling works)
    return list(best_h)
