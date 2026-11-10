import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Compute all pairwise sums for X1 + X2
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    # Flatten and sort
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # 2. Compute P = sum_{k,l} w_k w_l F(2*s_l - s_k)
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find cumulative weight of pair_sums < threshold
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def _objective_joint(params, n):
    """
    Objective function for joint optimization of support and weights.
    params: [s_1, s_2, ..., s_n, log_w_1, log_w_2, ..., log_w_n]
    Returns -P (to minimize)
    """
    s_raw = params[:n]
    log_w = params[n:]
    
    # Support: use softplus to ensure non-negative
    s = np.log1p(np.exp(s_raw))
    
    # Weights: use softmax to ensure they sum to 1 and are positive
    w = softmax(log_w)
    
    P = _compute_P(s, w)
    return -P

def _objective_weights_only(weights, support):
    """
    Objective function for weight-only optimization.
    weights: raw weights (will be normalized)
    Returns -P
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0)
    w_sum = w.sum()
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    P = _compute_P(support, w)
    return -P

def _refine_support_weights(support, weights, max_iter=2000, ftol=1e-12):
    """
    Refine both support and weights using L-BFGS-B.
    Returns (support, weights, score)
    """
    n = len(support)
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Convert to internal parameters
    # Support: invert softplus: s = log(1 + exp(x)) => x = log(expm1(s))
    # For s close to 0, expm1(s) ~ s, so x ~ log(s)
    # For numerical stability, clip s to [1e-10, large]
    s_clipped = np.clip(s, 1e-10, 1e6)
    s_params = np.log(np.expm1(s_clipped))
    
    # Weights: convert to log-softmax space
    # w = softmax(log_w) => log_w = log(w) (up to a constant)
    w_clipped = np.clip(w, 1e-10, 1.0)
    log_w_params = np.log(w_clipped)
    
    x0 = np.concatenate([s_params, log_w_params])
    
    try:
        result = minimize(
            _objective_joint, x0, args=(n,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
        )
        
        s_opt_raw = result.x[:n]
        log_w_opt = result.x[n:]
        
        s_opt = np.log1p(np.exp(s_opt_raw))
        w_opt = softmax(log_w_opt)
        
        # Ensure validity
        s_opt = np.maximum(s_opt, 0.0)
        w_opt = np.maximum(w_opt, 0.0)
        w_sum = w_opt.sum()
        if w_sum <= 1e-12:
            return support, weights, _compute_P(support, weights)
        w_opt = w_opt / w_sum
        
        score = _compute_P(s_opt, w_opt)
        return s_opt, w_opt, score
        
    except Exception:
        return support, weights, _compute_P(support, weights)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    def update_best(s, w, score):
        nonlocal best_support, best_weights, best_score
        if score > best_score:
            best_support = s.copy()
            best_weights = w.copy()
            best_score = score
    
    # Strategy: Joint optimization of support and weights for various n
    # Focus on n=2 to n=8, with multiple random restarts
    
    np.random.seed(42)
    
    # Try different n values
    for n in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 60.0:
            break
        
        # Number of restarts based on n
        if n <= 3:
            n_restarts = 50
        elif n <= 5:
            n_restarts = 30
        else:
            n_restarts = 15
        
        for r in range(n_restarts):
            if time.time() > deadline - 60.0:
                break
            
            # Random initialization
            # Support: random in [0, 10]
            s_init = np.random.uniform(0.1, 10.0, n)
            # Weights: Dirichlet
            w_init = np.random.dirichlet(np.ones(n) * 0.5)
            
            # Refine
            s_refined, w_refined, score = _refine_support_weights(s_init, w_init, max_iter=500)
            update_best(s_refined, w_refined, score)
            
            # If we have a good result, try to refine further
            if score > best_score - 0.01:
                s_refined2, w_refined2, score2 = _refine_support_weights(s_refined, w_refined, max_iter=1000)
                update_best(s_refined2, w_refined2, score2)
    
    # Also try structured initializations
    # E.g., points clustered in specific regions
    
    # Try: two clusters - one low, one high
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 60.0:
            break
        
        for r in range(20):
            if time.time() > deadline - 60.0:
                break
            
            # Random cluster structure
            low_center = np.random.uniform(0, 2)
            high_center = np.random.uniform(5, 15)
            
            s_init = np.empty(n)
            w_init = np.empty(n)
            
            n_low = np.random.randint(1, n)
            n_high = n - n_low
            
            s_init[:n_low] = low_center + np.random.normal(0, 0.5, n_low)
            s_init[n_low:] = high_center + np.random.normal(0, 1.0, n_high)
            
            w_init[:n_low] = np.random.uniform(0.5, 1.5, n_low)
            w_init[n_low:] = np.random.uniform(0.5, 1.5, n_high)
            w_init = np.maximum(w_init, 0.01)
            w_init = w_init / w_init.sum()
            
            s_init = np.maximum(s_init, 0.0)
            
            s_refined, w_refined, score = _refine_support_weights(s_init, w_init, max_iter=500)
            update_best(s_refined, w_refined, score)
    
    # Try: uniform spacing with perturbation
    for n in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 60.0:
            break
        
        for r in range(10):
            if time.time() > deadline - 60.0:
                break
            
            s_max = np.random.uniform(5, 20)
            s_init = np.linspace(0.1, s_max, n)
            s_init = s_init + np.random.normal(0, 0.1, n)
            s_init = np.maximum(s_init, 0.0)
            
            w_init = np.random.dirichlet(np.ones(n) * 0.3)
            
            s_refined, w_refined, score = _refine_support_weights(s_init, w_init, max_iter=500)
            update_best(s_refined, w_refined, score)
    
    # Final refinement of the best solution found
    if best_support is not None and time.time() < deadline - 30.0:
        s_final, w_final, score_final = _refine_support_weights(
            best_support, best_weights, max_iter=2000, ftol=1e-14
        )
        update_best(s_final, w_final, score_final)
    
    # Fallback if no valid solution found
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
