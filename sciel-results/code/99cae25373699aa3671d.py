# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Exact P[X1+X2+X3 < 2*X4] using sorted pair-sums + searchsorted. O(n^2 log n).
    """
    s = np.asarray(support, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    w = w / w_sum
    
    # Compute all pairwise sums s_i + s_j and their probabilities w_i * w_j
    pair_sums = s[:, None] + s[None, :]
    pair_probs = w[:, None] * w[None, :]
    
    # Flatten and sort
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Now for each (k, l), threshold = 2*s_l - s_k
    # We need sum of probabilities where pair_sum < threshold
    # Use searchsorted to find the index, then take cum_probs[index-1]
    
    # Build all thresholds: for k in range(n), l in range(n): 2*support[l] - support[k]
    # This is an (n, n) array where [k, l] = 2*support[l] - support[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    
    # Use searchsorted: for each threshold, find insertion point
    # We need to do this for n^2 thresholds
    # To speed up, we can process in batches or use broadcasting
    
    # Let's process row by row (over k)
    P = 0.0
    for k in range(n):
        # thresholds for this k: 2*support - support[k], shape (n,)
        thresh_row = 2.0 * s - s[k]  # shape (n,)
        
        # For each threshold in thresh_row, find count of sorted_sums < threshold
        counts = np.searchsorted(sorted_sums, thresh_row, side='left')
        
        # Get cumulative probabilities: where count > 0, take cum_probs[count-1], else 0
        # Handle count=0 case
        inner_probs = np.zeros(n)
        mask = counts > 0
        if np.any(mask):
            inner_probs[mask] = cum_probs[counts[mask] - 1]
        
        # P contribution from this k: sum over l of w[k]*w[l]*inner_probs[l]
        P += w[k] * np.dot(w, inner_probs)
    
    return P

def _objective(params, n):
    """
    Objective function for scipy.optimize.
    params: [s_0, delta_1, ..., delta_{n-1}, alpha_0, ..., alpha_{n-1}]
    s_0 >= 0
    delta_i > 0 (ensures s_i > s_{i-1})
    alpha_i are log-weights
    """
    # Unpack parameters
    s0 = params[0]
    deltas = params[1:n]
    alphas = params[n:2*n]
    
    # Reconstruct support
    # s_0 = s0
    # s_1 = s0 + delta_1
    # ...
    # To enforce s_i >= 0, we can use softplus or just clip.
    # Since s0 is free, we want s0 >= 0.
    # To enforce s_i > s_{i-1}, we use deltas > 0.
    # We can use log-transformed deltas to enforce positivity.
    # Let delta_i = exp(params[1+i])? No, that makes them very small.
    # Better: just use the raw params and rely on the optimizer to stay in bounds?
    # Nelder-Mead doesn't handle bounds well.
    # Let's use a transformation:
    # s0 = softplus(params[0])
    # delta_i = softplus(params[1+i])
    # This ensures s0 > 0 and deltas > 0.
    
    s0_val = np.logaddexp(0, params[0]) # softplus
    deltas_val = np.array([np.logaddexp(0, d) for d in deltas])
    
    support = np.zeros(n)
    support[0] = s0_val
    for i in range(1, n):
        support[i] = support[i-1] + deltas_val[i-1]
    
    # Weights from alphas
    # w_i = exp(alpha_i) / sum(exp(alpha_j))
    # To avoid overflow, subtract max
    max_alpha = np.max(alphas)
    exp_alphas = np.exp(alphas - max_alpha)
    weights = exp_alphas / np.sum(exp_alphas)
    
    # Compute P
    P = _compute_P_fast(support, weights)
    
    # We want to maximize P, so minimize -P
    return -P

def _optimize_fixed_n(n, initial_support, initial_weights, time_limit):
    """
    Optimize an n-point distribution starting from initial_support/weights.
    """
    # Parameterize
    s0 = initial_support[0]
    deltas = np.diff(initial_support)
    # Ensure deltas are positive
    deltas = np.maximum(deltas, 1e-6)
    
    # Inverse softplus to get params
    # x = log(exp(y) - 1)
    def inv_softplus(y):
        if y < 1e-6:
            return np.log(y)
        return np.log(np.expm1(y))
    
    p_s0 = inv_softplus(s0)
    p_deltas = np.array([inv_softplus(d) for d in deltas])
    
    # Log weights
    log_weights = np.log(initial_weights)
    
    params = np.concatenate(([p_s0], p_deltas, log_weights))
    
    start_time = time.time()
    
    def timeout_check():
        return time.time() - start_time > time_limit
    
    # Use Nelder-Mead
    # Note: Nelder-Mead can be slow for high dimensions. 
    # For n=10, dim=20. 
    # We will run it for a limited number of iterations or time.
    
    # To handle the non-smoothness, we might need to run multiple restarts or use a more robust method.
    # However, the objective is "almost" smooth except for the jumps.
    
    best_P = _compute_P_fast(initial_support, initial_weights)
    best_support = initial_support.copy()
    best_weights = initial_weights.copy()
    
    # Run optimization
    # maxiter is a rough guide, we also check time
    maxiter = 1000
    
    def callback(xk):
        # Check time
        if timeout_check():
            return True # Stop
        
        # Evaluate current
        # Reconstruct support and weights from xk
        s0_val = np.logaddexp(0, xk[0])
        deltas_val = np.array([np.logaddexp(0, d) for d in xk[1:n]])
        support = np.zeros(n)
        support[0] = s0_val
        for i in range(1, n):
            support[i] = support[i-1] + deltas_val[i-1]
        
        alphas = xk[n:2*n]
        max_alpha = np.max(alphas)
        exp_alphas = np.exp(alphas - max_alpha)
        weights = exp_alphas / np.sum(exp_alphas)
        
        P = _compute_P_fast(support, weights)
        
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
            
        return False # Continue
    
    try:
        result = minimize(
            _objective, 
            params, 
            args=(n,), 
            method='Nelder-Mead', 
            options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False}
        )
    except Exception:
        pass
    
    return best_support, best_weights, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    global_start = time.time()
    total_time_limit = 620  # Leave 10s margin
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Candidate 1: Baseline uniform
    n_base = 10
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    P_base = _compute_P_fast(support_base, weights_base)
    best_P = P_base
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    
    # Candidate 2: Pareto-like tail
    # s_i = i^alpha, w_i proportional to 1/i^beta
    # Try a few configurations
    
    for n in [5, 8, 10, 15, 20]:
        if time.time() - global_start > total_time_limit - 60:
            break
            
        # Generate initial guess: Pareto tail
        # s_i = (i+1)^gamma
        # w_i = 1 / (i+1)^delta
        gamma = 1.5
        delta = 1.2
        
        support_init = np.array([(i+1)**gamma for i in range(n)])
        support_init[0] = 0.0 # Force one point at 0? Or keep it?
        # Actually, mass at 0 is often beneficial.
        # Let's try a mixture: 50% at 0, 50% on Pareto tail
        # But our optimizer handles general support.
        # Let's just use the Pareto tail as initial guess.
        
        weights_init = np.array([1.0 / ((i+1)**delta) for i in range(n)])
        weights_init = weights_init / weights_init.sum()
        
        P_init = _compute_P_fast(support_init, weights_init)
        if P_init > best_P:
            best_P = P_init
            best_support = support_init.copy()
            best_weights = weights_init.copy()
            
        # Optimize this candidate
        time_remaining = total_time_limit - (time.time() - global_start) - 10
        if time_remaining < 5:
            continue
            
        # Allocate time based on n
        # Smaller n can be optimized longer
        alloc_time = min(60, time_remaining * (10.0 / (n + 10)))
        
        opt_support, opt_weights, opt_P = _optimize_fixed_n(
            n, support_init, weights_init, alloc_time
        )
        
        if opt_P > best_P:
            best_P = opt_P
            best_support = opt_support.copy()
            best_weights = opt_weights.copy()
            
    # If we have time, try to refine the best found solution with a smaller n?
    # Or just return the best.
    
    # Final validation
    if best_support is None or best_weights is None:
        # Fallback
        best_support = np.array([0.0, 1.0, 2.0, 4.0])
        best_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / best_weights.sum()
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
