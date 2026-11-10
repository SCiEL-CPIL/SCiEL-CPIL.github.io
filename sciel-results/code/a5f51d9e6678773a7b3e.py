# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _calc_p_fast(support, weights):
    """
    Computes P(X1+X2+X3 < 2*X4) exactly.
    support, weights: 1D arrays of same length n.
    """
    w = weights / np.sum(weights)
    n = len(support)
    
    # Precompute pair sums and weights
    # s_i + s_j
    # We can use broadcasting
    s_i = support[:, np.newaxis]
    s_j = support[np.newaxis, :]
    pair_sums = s_i + s_j
    pair_weights = w[:, np.newaxis] * w[np.newaxis, :]
    
    # Flatten
    ps_flat = pair_sums.flatten()
    pw_flat = pair_weights.flatten()
    
    # Sort pair sums to allow binary search for cumulative sums
    sort_idx = np.argsort(ps_flat)
    ps_sorted = ps_flat[sort_idx]
    pw_sorted = pw_flat[sort_idx]
    cum_pw = np.cumsum(pw_sorted)
    
    total_p = 0.0
    for l in range(n):
        target = 2.0 * support[l]
        # We need sum_{i,j,k} w_i w_j w_k I(s_i+s_j+s_k < target)
        # = sum_k w_k ( sum_{i,j} w_i w_j I(s_i+s_j < target - s_k) )
        
        inner_sum = 0.0
        for k in range(n):
            limit = target - support[k]
            # Find index of last element < limit in ps_sorted
            # Strict inequality: s_i+s_j < limit
            idx = np.searchsorted(ps_sorted, limit, side='left')
            if idx > 0:
                inner_sum += w[k] * cum_pw[idx-1]
            # else 0
        
        total_p += w[l] * inner_sum
        
    return total_p

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 500.0 # Safety margin
    
    # Baseline
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    p_base = _calc_p_fast(support_base, weights_base)
    
    best_p = p_base
    best_support = support_base
    best_weights = weights_base
    
    # Optimization on smaller support
    n_opt = 16
    support_opt = np.arange(n_opt, dtype=np.float64)
    
    def objective(v):
        w = np.exp(v - np.max(v)) # Stable softmax
        w = w / np.sum(w)
        return -_calc_p_fast(support_opt, w)
    
    # Initial guesses
    inits = [
        np.zeros(n_opt), # Uniform
        np.arange(n_opt) * -0.1, # Geometric-ish
        np.array([0.5] + [0.0]*(n_opt-2) + [0.5]) # Bimodal (needs softmax input adjustment)
    ]
    # Adjust bimodal for softmax: high values for 0 and n-1
    inits[2] = np.zeros(n_opt)
    inits[2][0] = 5.0
    inits[2][-1] = 5.0
    
    for v0 in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, v0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success:
                w_opt = np.exp(res.x - np.max(res.x))
                w_opt = w_opt / np.sum(w_opt)
                p_opt = _calc_p_fast(support_opt, w_opt)
                if p_opt > best_p:
                    best_p = p_opt
                    best_support = support_opt.copy()
                    best_weights = w_opt.copy()
        except Exception:
            pass

    # Check time
    if time.time() > deadline:
        return best_support, best_weights

    return best_support, best_weights

# EVOLVE-BLOCK-END
