# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using O(n^2 log n) searchsorted.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)

def _objective(params, n_points):
    """
    Objective function for gradient-based optimization.
    params[0] = log_alpha (for log-uniform support spread)
    params[1] = log_beta (for exponential tail support)
    params[2] = log_mix (mixing weight between log-uniform and exponential)
    params[3:] = log_weights (length n_points)
    
    We define the support s_i as a deterministic function of the first 3 params.
    Then we optimize the weights.
    """
    try:
        log_alpha = params[0]
        log_beta = params[1]
        log_mix = params[2]
        log_w = params[3:3+n_points]
        
        # Unpack and constrain parameters
        alpha = np.exp(np.clip(log_alpha, -10, 10)) # Scale for log-uniform
        beta = np.exp(np.clip(log_beta, -10, 10))  # Rate for exponential
        mix = 1.0 / (1.0 + np.exp(-np.clip(log_mix, -10, 10))) # Sigmoid mix
        
        # Generate support points
        # We want a mix of a Log-Uniform distribution and an Exponential distribution.
        # Let's define the support points as quantiles of this mixture.
        # To keep it simple and smooth, we define s_i based on i/n.
        u = np.linspace(1e-6, 1-1e-6, n_points)
        
        # Log-uniform CDF inverse: s = exp(log(low) + u * (log(high) - log(low)))
        # Let's fix low=0.1, high=10.0 for log-uniform part.
        # s_log = 0.1 * (10.0/0.1)^u = 0.1 * 100^u = exp(log(0.1) + u * log(100))
        s_log_part = np.exp(np.log(0.1) + u * np.log(100.0))
        
        # Exponential CDF inverse: s = -log(1-u)/beta
        s_exp_part = -np.log(1 - u) / beta
        
        # Mixture
        s_support = mix * s_log_part + (1 - mix) * s_exp_part
        
        # Ensure support is strictly positive and sorted
        s_support = np.maximum(s_support, 1e-9)
        s_support = np.sort(s_support)
        
        # Unpack weights
        # Normalize log weights to ensure sum(exp(log_w)) = 1
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w_sum = np.sum(exp_w)
        if w_sum < 1e-300:
            return 1e6
        w = exp_w / w_sum
        
        if np.any(~np.isfinite(w)) or np.any(~np.isfinite(s_support)):
            return 1e6
        
        P = _compute_P_fast(s_support, w)
        
        if not np.isfinite(P):
            return 1e6
            
        return -P
    except Exception:
        return 1e6

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Gradient-based optimization (L-BFGS-B) on a parameterized family
    of supports (mixture of Log-Uniform and Exponential quantiles) and weights.
    """
    t_start = time.time()
    deadline = t_start + 590.0 # Safety margin
    
    # We will try a few different n_points and initializations
    # and keep the best result.
    
    best_s = np.array([0.0, 1.0])
    best_w = np.array([0.5, 0.5])
    best_P = _compute_P_fast(best_s, best_w)
    
    # Candidate n_points
    n_candidates = [3, 5, 7, 10]
    
    # Initial parameters:
    # log_alpha, log_beta, log_mix, log_w...
    # Let's start with a mix of 0.5 (log_mix=0), alpha=1 (log_alpha=0), beta=1 (log_beta=0)
    # Weights start uniform (log_w=0)
    
    rng = np.random.default_rng(42)
    
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this n
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize params
            # log_alpha ~ U(-2, 2)
            # log_beta ~ U(-2, 2)
            # log_mix ~ U(-2, 2)
            # log_w ~ U(-2, 2)
            init_params = np.zeros(3 + n)
            init_params[0] = rng.uniform(-2, 2)
            init_params[1] = rng.uniform(-2, 2)
            init_params[2] = rng.uniform(-2, 2)
            init_params[3:] = rng.uniform(-2, 2)
            
            # Optimize
            try:
                res = minimize(
                    _objective,
                    init_params,
                    args=(n,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e5:
                # Extract best params
                best_params = res.x
                log_alpha = best_params[0]
                log_beta = best_params[1]
                log_mix = best_params[2]
                log_w = best_params[3:3+n]
                
                alpha = np.exp(np.clip(log_alpha, -10, 10))
                beta = np.exp(np.clip(log_beta, -10, 10))
                mix = 1.0 / (1.0 + np.exp(-np.clip(log_mix, -10, 10)))
                
                u = np.linspace(1e-6, 1-1e-6, n)
                s_log_part = np.exp(np.log(0.1) + u * np.log(100.0))
                s_exp_part = -np.log(1 - u) / beta
                s_support = mix * s_log_part + (1 - mix) * s_exp_part
                s_support = np.maximum(s_support, 1e-9)
                s_support = np.sort(s_support)
                
                max_log_w = np.max(log_w)
                exp_w = np.exp(log_w - max_log_w)
                w_sum = np.sum(exp_w)
                if w_sum > 1e-9:
                    w = exp_w / w_sum
                else:
                    w = np.ones(n) / n
                    
                if np.all(np.isfinite(s_support)) and np.all(np.isfinite(w)):
                    P_val = _compute_P_fast(s_support, w)
                    if P_val > best_P:
                        best_P = P_val
                        best_s = s_support.copy()
                        best_w = w.copy()
                        
    # Final validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
