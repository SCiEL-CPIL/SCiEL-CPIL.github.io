# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_prob(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly.
    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i w_j)
    """
    n = len(support)
    # Create all pairwise sums for X1, X2
    # s_i + s_j
    S = support[:, None] + support[None, :]
    W = weights[:, None] * weights[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Sort S_flat and compute cumulative sum of W_flat
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    
    total_prob = 0.0
    # Iterate over X3 (index k) and X4 (index l)
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Find number of pairs (i,j) with S_ij < threshold
            # np.searchsorted returns the index of the first element >= threshold
            # So we want sum of W_sorted[0:idx]
            idx = np.searchsorted(S_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = W_cumsum[idx - 1]
            else:
                prob_inner = 0.0
            
            total_prob += weights[k] * weights[l] * prob_inner
            
    return total_prob

def _objective(params):
    """
    Minimize negative probability.
    params: [p0, alpha, m]
    """
    p0, alpha, m = params
    # Clip parameters to valid ranges
    p0 = np.clip(p0, 0.01, 0.99)
    alpha = np.clip(alpha, 0.5, 5.0)
    m = np.clip(m, 0.1, 10.0)
    
    # Construct support and weights
    # 1 point at 0
    # M points in exponential tail
    M = 20
    support = np.zeros(M + 1)
    weights = np.zeros(M + 1)
    
    support[0] = 0.0
    weights[0] = p0
    
    # Tail points: m * 2^i
    for i in range(M):
        support[i+1] = m * (2.0 ** i)
        # Weight proportional to x^-alpha
        weights[i+1] = (support[i+1]) ** (-alpha)
        
    # Normalize tail weights to sum to 1 - p0
    tail_sum = np.sum(weights[1:])
    if tail_sum > 1e-9:
        weights[1:] *= (1.0 - p0) / tail_sum
    else:
        # Fallback if weights are zero
        weights[1:] = (1.0 - p0) / M
        support[1:] = np.linspace(m, m*100, M)

    # Validate
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        return 0.0
        
    prob = _compute_prob(support, weights)
    return -prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Approach: Mixture of point mass at 0 and discrete Pareto tail.
    Optimize parameters using Nelder-Mead.
    """
    # Initial guess
    # p0: prob at 0, alpha: pareto shape, m: scale
    x0 = np.array([0.5, 1.5, 1.0])
    
    # Bounds are handled by clipping in objective, but Nelder-Mead doesn't support bounds directly
    # We rely on clipping and good initial guess.
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        best_params = res.x
    except Exception:
        best_params = x0
        
    p0, alpha, m = best_params
    p0 = np.clip(p0, 0.01, 0.99)
    alpha = np.clip(alpha, 0.5, 5.0)
    m = np.clip(m, 0.1, 10.0)
    
    M = 20
    support = np.zeros(M + 1)
    weights = np.zeros(M + 1)
    
    support[0] = 0.0
    weights[0] = p0
    
    for i in range(M):
        support[i+1] = m * (2.0 ** i)
        weights[i+1] = (support[i+1]) ** (-alpha)
        
    tail_sum = np.sum(weights[1:])
    if tail_sum > 1e-9:
        weights[1:] *= (1.0 - p0) / tail_sum
    else:
        weights[1:] = (1.0 - p0) / M
        support[1:] = np.linspace(m, m*100, M)

    # Final sanity check
    if np.any(weights < -1e-9) or np.any(support < -1e-9) or np.sum(weights) <= 1e-9:
        # Fallback to a known good distribution if optimization failed badly
        # e.g. uniform on 0..10
        n_fallback = 11
        support = np.arange(n_fallback, dtype=np.float64)
        weights = np.ones(n_fallback, dtype=np.float64)
        
    return support, weights

# EVOLVE-BLOCK-END
