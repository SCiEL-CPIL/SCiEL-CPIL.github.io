# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete distribution.
    Uses sorted pairwise sums with cumulative weights for O(n^2 log n) per call.
    support: np.array of shape (n,)
    weights: np.array of shape (n,), assumed normalized
    """
    s = support
    w = weights
    n = len(s)
    
    # Precompute pairwise sums and product weights
    sums = s[:, None] + s[None, :]
    w_prod = w[:, None] * w[None, :]
    
    # Flatten and sort for fast threshold queries
    sums_flat = sums.ravel()
    w_prod_flat = w_prod.ravel()
    sort_idx = np.argsort(sums_flat)
    sums_sorted = sums_flat[sort_idx]
    w_prod_sorted = w_prod_flat[sort_idx]
    cumsum_w = np.zeros(n * n + 1)
    cumsum_w[1:] = np.cumsum(w_prod_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Count pairs with sum < threshold (strict inequality)
            idx = np.searchsorted(sums_sorted, threshold, side='left')
            inner_sum = cumsum_w[idx]
            P += w[k] * w[l] * inner_sum
            
    return P

def _neg_P_from_params(params, n):
    """
    Objective function that takes raw parameters and returns -P.
    params: [support_1, ..., support_{n-1}, log_w_1, ..., log_w_n]
    Support point 0 is fixed at 0 (scale invariance).
    Support points are enforced to be non-negative and sorted.
    Weights are softmax-normalized.
    """
    # Extract support (first n-1 elements, s[0]=0 fixed)
    s_raw = params[:n-1]
    # Enforce non-negative and increasing: s[i] = max(0, s[i-1] + max(0, s_raw[i]))
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i-1] + max(0.0, s_raw[i-1])
    
    # Extract weights via softmax
    log_w = params[n-1:]
    # Shift for numerical stability
    log_w_shifted = log_w - np.max(log_w)
    w = np.exp(log_w_shifted)
    w = w / np.sum(w)
    
    return -_calc_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0])
    best_weights = np.array([1.0])
    
    # Try small n values with joint optimization of support and weights
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        
        # Number of restarts: more for smaller n since each optimization is cheaper
        remaining = deadline - time.time()
        if n <= 3:
            num_restarts = min(200, int(remaining / 0.05))
        elif n <= 5:
            num_restarts = min(100, int(remaining / 0.1))
        elif n <= 7:
            num_restarts = min(50, int(remaining / 0.2))
        else:
            num_restarts = min(30, int(remaining / 0.5))
        
        num_restarts = max(num_restarts, 5)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Support: start with increasing positive values
            s_init = np.sort(np.random.exponential(1.0, size=n-1))
            # Log-weights: random
            log_w_init = np.random.normal(0, 1, size=n)
            
            params_init = np.concatenate([s_init, log_w_init])
            
            try:
                res = so.minimize(
                    _neg_P_from_params,
                    params_init,
                    args=(n,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if res.fun < 0:  # P > 0
                    # Recover support and weights
                    s_opt = np.zeros(n)
                    for i in range(1, n):
                        s_opt[i] = s_opt[i-1] + max(0.0, res.x[i-1])
                    
                    log_w_opt = res.x[n-1:]
                    log_w_shifted = log_w_opt - np.max(log_w_opt)
                    w_opt = np.exp(log_w_shifted)
                    w_opt = w_opt / np.sum(w_opt)
                    
                    P_val = _calc_P(s_opt, w_opt)
                    
                    if P_val > best_P:
                        best_P = P_val
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
                        
            except Exception:
                pass
    
    # If no optimization succeeded, fall back to a known good distribution
    if best_P <= 0.0:
        # Use a distribution that puts mass on 0 and 1
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
