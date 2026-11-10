import numpy as np
import time
import sys
from scipy.optimize import minimize
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
    
    # Normalize weights just in case, though optimization should keep them normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Compute all pairwise sums for X1 + X2
    # s_i + s_j
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
    # Thresholds matrix T[k, l] = 2*s_l - s_k
    # Note: s is (n,).
    # T shape (n, n) where T[k, l] corresponds to X3=s_k, X4=s_l
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find cumulative weight of pair_sums < threshold
    # searchsorted returns index of first element >= x.
    # We want < x, so index is count of elements < x.
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative weights
    # If idx > 0, cum_weight is cum_weights[idx-1]
    # If idx == 0, cum_weight is 0
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    # w[:, None] is (n, 1), w[None, :] is (1, n). Product is (n, n)
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def _compute_P_gradient(support, weights):
    """
    Compute P and its gradient w.r.t. weights for a discrete measure.
    Returns (P, grad_w) where grad_w has same shape as weights.
    """
    n = len(support)
    if n == 0:
        return 0.0, np.zeros(0)
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0, np.zeros(n)
    w = w / total_w
    
    # Pairwise sums and weights
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    thresholds = 2.0 * s[None, :] - s[:, None]
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    # Gradient w.r.t. w: 
    # P = sum_{k,l} w_k w_l probs[k,l]
    # dP/dw_m = sum_{k,l} (delta_{m,k} w_l + w_k delta_{m,l}) probs[k,l]
    #         = sum_l w_l probs[m,l] + sum_k w_k probs[k,m]
    #         = (w @ probs)[m] + (probs @ w)[m]
    grad_w = (w @ probs) + (probs @ w)
    
    return float(P), grad_w

def _project_to_simplex(v):
    """Project vector v onto the probability simplex (sum=1, all >= 0)."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w_sum = w.sum()
    if w_sum > 1e-12:
        w = w / w_sum
    else:
        w = np.ones(n) / n
    return w

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
    
    # Strategy: Use a fixed grid of support points and optimize weights via projected gradient
    # This avoids the combinatorial search over n and local minima traps
    
    # Try different grid sizes and ranges
    grid_configs = [
        # (n_points, min_val, max_val)
        (64, 0.0, 5.0),
        (128, 0.0, 5.0),
        (256, 0.0, 5.0),
        (128, 0.0, 10.0),
        (64, 0.0, 1.0),
        (128, 0.0, 1.0),
    ]
    
    for n_grid, s_min, s_max in grid_configs:
        if time.time() > deadline - 30.0:
            break
        
        # Create grid
        support_grid = np.linspace(s_min, s_max, n_grid)
        
        # Number of projected gradient steps
        n_steps = 2000 if n_grid <= 128 else 1000
        
        # Learning rate schedule
        base_lr = 0.1
        
        # Multiple random restarts
        n_restarts = 5 if n_grid <= 128 else 3
        
        for r in range(n_restarts):
            if time.time() > deadline - 30.0:
                break
            
            # Random initial weights
            if r == 0:
                # Uniform initialization
                w = np.ones(n_grid) / n_grid
            else:
                w = np.random.dirichlet(np.ones(n_grid) * 0.5)
            
            current_w = w.copy()
            current_score = _compute_P(support_grid, current_w)
            update_best(support_grid, current_w, current_score)
            
            # Projected gradient ascent
            for step in range(n_steps):
                if time.time() > deadline - 30.0:
                    break
                
                # Compute P and gradient
                P_val, grad_w = _compute_P_gradient(support_grid, current_w)
                
                # Learning rate decay
                lr = base_lr / (1.0 + step * 0.001)
                
                # Gradient ascent step
                new_w = current_w + lr * grad_w
                
                # Project to simplex
                new_w = _project_to_simplex(new_w)
                
                # Check for improvement
                new_score = _compute_P(support_grid, new_w)
                
                if new_score > current_score:
                    current_w = new_w
                    current_score = new_score
                    update_best(support_grid, current_w, current_score)
                else:
                    # Reduce learning rate if no improvement
                    lr *= 0.5
                    new_w = current_w + lr * grad_w
                    new_w = _project_to_simplex(new_w)
                    new_score = _compute_P(support_grid, new_w)
                    if new_score > current_score:
                        current_w = new_w
                        current_score = new_score
                        update_best(support_grid, current_w, current_score)
            
            # Try a few more restarts with different initializations
            # Use the best found so far as a starting point
            if best_weights is not None and len(best_weights) == n_grid:
                w = best_weights.copy()
                current_w = w.copy()
                current_score = _compute_P(support_grid, current_w)
                
                for step in range(n_steps // 2):
                    if time.time() > deadline - 30.0:
                        break
                    
                    P_val, grad_w = _compute_P_gradient(support_grid, current_w)
                    lr = base_lr * 0.5 / (1.0 + step * 0.001)
                    new_w = current_w + lr * grad_w
                    new_w = _project_to_simplex(new_w)
                    new_score = _compute_P(support_grid, new_w)
                    
                    if new_score > current_score:
                        current_w = new_w
                        current_score = new_score
                        update_best(support_grid, current_w, current_score)
    
    # Also try small-n local search as fallback/complement
    if time.time() < deadline - 60.0:
        for n in [2, 3, 4, 5]:
            if time.time() > deadline - 60.0:
                break
            
            n_restarts = 30 if n <= 3 else 15
            
            for r in range(n_restarts):
                if time.time() > deadline - 60.0:
                    break
                
                s_init = np.random.uniform(0, 5, n)
                w_init = np.random.dirichlet(np.ones(n))
                
                def objective(params):
                    s_opt = np.asarray(params[:n], dtype=np.float64)
                    log_w_opt = np.asarray(params[n:], dtype=np.float64)
                    s_pos = np.log1p(np.exp(s_opt))
                    w_opt = softmax(log_w_opt)
                    P = _compute_P(s_pos, w_opt)
                    return -P
                
                s_params_init = np.log(np.expm1(s_init + 1e-6))
                w_params_init = np.log(w_init + 1e-6)
                x0 = np.concatenate([s_params_init, w_params_init])
                
                try:
                    result = minimize(objective, x0, method='L-BFGS-B',
                                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-7})
                    
                    if result.success or result.fun < objective(x0):
                        s_opt = result.x[:n]
                        log_w_opt = result.x[n:]
                        s_final = np.log1p(np.exp(s_opt))
                        w_final = softmax(log_w_opt)
                        
                        if np.any(s_final < -1e-9) or np.any(w_final < -1e-9):
                            continue
                        if w_final.sum() <= 1e-9:
                            continue
                            
                        score = _compute_P(s_final, w_final)
                        update_best(s_final, w_final, score)
                        
                except Exception:
                    pass
    
    # Fallback if no valid solution found
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
