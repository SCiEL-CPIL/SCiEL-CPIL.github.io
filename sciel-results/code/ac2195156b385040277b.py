# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 600.0  # wall-clock limit for the evaluator
_SAFETY_MARGIN = 30.0  # seconds to leave before the limit


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for i.i.d. discrete variables with given
    support and (unnormalized) weights.
    """
    w = weights / weights.sum()
    s = support
    n = len(s)
    # Pairwise sums: S[i,j] = s[i] + s[j]
    S = s[:, None] + s[None, :]  # n x n
    # Thresholds: T[k,l] = 2*s[l] - s[k]
    T = 2.0 * s[None, :] - s[:, None]  # n x n
    
    # Build W[i,j] = w[i]*w[j]
    W = np.outer(w, w)  # n x n
    
    S_flat = S.ravel()
    W_flat = W.ravel()
    
    # Sort S_flat for efficient counting
    order = np.argsort(S_flat)
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cumsum = np.cumsum(W_sorted)
    
    # For each threshold T[k,l], find number of S < T using binary search
    T_flat = T.ravel()
    counts = np.searchsorted(S_sorted, T_flat, side='left')
    inner = np.zeros_like(T_flat, dtype=np.float64)
    mask = counts > 0
    inner[mask] = W_cumsum[counts[mask] - 1]
    inner[~mask] = 0.0
    
    inner_2d = inner.reshape(n, n)
    P = np.sum(np.outer(w, w) * inner_2d)
    return P


def _neg_P(params, n):
    """
    Negative objective for optimization.
    params: 2n-1 dimensional vector: [s_1, ..., s_n, w_1, ..., w_{n-1}]
    w_n is determined by 1 - sum(w_1..w_{n-1})
    """
    s = params[:n]
    w_rest = params[n:]
    w_n_val = 1.0 - np.sum(w_rest)
    
    # Check constraints: s_i >= 0, w_i >= 0
    if np.any(s < -1e-9) or np.any(w_rest < -1e-9) or w_n_val < -1e-9:
        return 1e10
    
    w = np.concatenate([w_rest, [w_n_val]])
    
    # Ensure support is non-negative and has some spread
    if np.max(s) < 1e-9:
        return 1e10
    
    P = _compute_P(s, w)
    return -P


def _neg_P_log(params, n):
    """
    Negative objective with log-parameterization for support and softmax for weights.
    params: 2n-1 dimensional vector: [log_s_1, ..., log_s_n, log_w_1, ..., log_w_{n-1}]
    w_n is determined by softmax normalization.
    """
    log_s = params[:n]
    log_w_rest = params[n:]
    
    s = np.exp(log_s)
    
    # Softmax for weights
    log_w_all = np.concatenate([log_w_rest, [0.0]])
    log_w_max = np.max(log_w_all)
    w_unnorm = np.exp(log_w_all - log_w_max)
    w = w_unnorm / np.sum(w_unnorm)
    
    if np.max(s) < 1e-9:
        return 1e10
    
    P = _compute_P(s, w)
    return -P


def _try_n_points(n, max_time_per_try=30.0):
    """
    Try to optimize an n-point discrete measure using local optimization.
    Returns (support, weights, P) for the best found.
    """
    start_time = time.time()
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # Try multiple random initializations
    n_init = 8
    for init_i in range(n_init):
        if time.time() - start_time > max_time_per_try:
            break
        if time.time() > _START_TIME + _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        # Random initialization
        if init_i == 0:
            # Start with uniform-ish
            s = np.linspace(0.01, 1, n)
            w = np.ones(n) / n
        elif init_i == 1:
            # Geometric-like with wide range
            s = np.geomspace(0.001, 10, n)
            w = np.ones(n) / n
        elif init_i == 2:
            # Heavy-tail: small values with high weight, large values with low weight
            s = np.geomspace(0.01, 100, n)
            w = np.exp(-np.linspace(0, 3, n))
            w = w / np.sum(w)
        elif init_i == 3:
            # Bimodal: cluster near 0 and cluster at larger values
            half = n // 2
            s = np.concatenate([np.linspace(0.001, 0.1, half), np.linspace(1, 10, n - half)])
            w = np.concatenate([np.ones(half) * 0.7, np.ones(n - half) * 0.3])
            w = w / np.sum(w)
        else:
            # Random geometric with random ratio
            ratio = np.exp(np.random.uniform(0.5, 2.0))
            s = np.geomspace(0.01, 0.01 * ratio**(n-1), n)
            w = np.random.dirichlet(np.ones(n))
        
        # Use log-parameterization for smoother optimization
        log_s = np.log(np.maximum(s, 1e-6))
        log_w_rest = np.log(np.maximum(w[:-1], 1e-6))
        params = np.concatenate([log_s, log_w_rest])
        
        result = minimize(_neg_P_log, params, args=(n,), method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-12})
        
        if result.fun < -best_P:
            log_s_opt = result.x[:n]
            log_w_rest_opt = result.x[n:]
            
            s_opt = np.exp(log_s_opt)
            log_w_all = np.concatenate([log_w_rest_opt, [0.0]])
            log_w_max = np.max(log_w_all)
            w_unnorm = np.exp(log_w_all - log_w_max)
            w_opt = w_unnorm / np.sum(w_unnorm)
            
            if np.any(s_opt < 0) or np.any(w_opt < 0) or np.any(np.isnan(s_opt)) or np.any(np.isnan(w_opt)):
                continue
            if np.sum(w_opt) < 1e-9:
                continue
            
            s_opt = np.maximum(s_opt, 0.0)
            w_opt = np.maximum(w_opt, 0.0)
            P_val = _compute_P(s_opt, w_opt)
            if P_val > best_P:
                best_P = P_val
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
    
    return best_support, best_weights, best_P


def _try_n_points_powell(n, max_time_per_try=30.0):
    """
    Try to optimize an n-point discrete measure using Powell's method.
    Returns (support, weights, P) for the best found.
    """
    start_time = time.time()
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # Try a few initializations
    n_init = 3
    for init_i in range(n_init):
        if time.time() - start_time > max_time_per_try:
            break
        if time.time() > _START_TIME + _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        if init_i == 0:
            s = np.linspace(0.01, 1, n)
            w = np.ones(n) / n
        elif init_i == 1:
            s = np.geomspace(0.001, 10, n)
            w = np.ones(n) / n
        else:
            s = np.geomspace(0.01, 100, n)
            w = np.exp(-np.linspace(0, 3, n))
            w = w / np.sum(w)
        
        log_s = np.log(np.maximum(s, 1e-6))
        log_w_rest = np.log(np.maximum(w[:-1], 1e-6))
        params = np.concatenate([log_s, log_w_rest])
        
        result = minimize(_neg_P_log, params, args=(n,), method='Powell',
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12})
        
        if result.fun < -best_P:
            log_s_opt = result.x[:n]
            log_w_rest_opt = result.x[n:]
            
            s_opt = np.exp(log_s_opt)
            log_w_all = np.concatenate([log_w_rest_opt, [0.0]])
            log_w_max = np.max(log_w_all)
            w_unnorm = np.exp(log_w_all - log_w_max)
            w_opt = w_unnorm / np.sum(w_unnorm)
            
            if np.any(s_opt < 0) or np.any(w_opt < 0) or np.any(np.isnan(s_opt)) or np.any(np.isnan(w_opt)):
                continue
            if np.sum(w_opt) < 1e-9:
                continue
            
            s_opt = np.maximum(s_opt, 0.0)
            w_opt = np.maximum(w_opt, 0.0)
            P_val = _compute_P(s_opt, w_opt)
            if P_val > best_P:
                best_P = P_val
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
    
    return best_support, best_weights, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    # Start with the baseline
    n_baseline = 32
    support_baseline = np.arange(n_baseline, dtype=np.float64)
    weights_baseline = np.ones(n_baseline, dtype=np.float64) / n_baseline
    P_baseline = _compute_P(support_baseline, weights_baseline)
    
    best_support = support_baseline
    best_weights = weights_baseline
    best_P = P_baseline
    
    # Try optimizing with various n using Nelder-Mead
    for n in [3, 4, 5, 6, 8]:
        if time.time() > _START_TIME + _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        support_opt, weights_opt, P_opt = _try_n_points(n)
        if support_opt is not None and P_opt > best_P:
            best_support = support_opt
            best_weights = weights_opt
            best_P = P_opt
    
    # Try Powell's method for potentially better local optima
    for n in [4, 5, 6]:
        if time.time() > _START_TIME + _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        support_opt, weights_opt, P_opt = _try_n_points_powell(n)
        if support_opt is not None and P_opt > best_P:
            best_support = support_opt
            best_weights = weights_opt
            best_P = P_opt
    
    # Final validation
    support_out = np.maximum(best_support, 0.0)
    weights_out = np.maximum(best_weights, 0.0)
    
    # Ensure no NaN or inf
    if np.any(np.isnan(support_out)) or np.any(np.isinf(support_out)) or \
       np.any(np.isnan(weights_out)) or np.any(np.isinf(weights_out)):
        return support_baseline, weights_baseline
    
    if len(support_out) != len(weights_out) or len(support_out) < 1:
        return support_baseline, weights_baseline
    
    if np.sum(weights_out) <= 1e-9:
        return support_baseline, weights_baseline
    
    return support_out, weights_out


# EVOLVE-BLOCK-END
