import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# The problem implies a fixed N from the seed program context.
# Seed used N=100.
N = 100

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _soft_min_obj_and_grad(params, beta):
    """
    Objective and gradient for symmetric h.
    params: length N/2, represents h[0:N/2]
    h is constructed symmetrically.
    """
    n_half = len(params)
    n = 2 * n_half
    # Construct full h
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    corrs = _compute_overlaps(h)
    
    # Soft min of corrs
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    # d(soft_min)/d(corrs_j) = exp(-beta * corrs_j) / sum(exp(-beta * corrs))
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg
    
    # d(soft_min)/d(h) = sum_j weights_j * d(corrs_j)/d(h)
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w == 0: continue
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = - grad_h
    grad_h = -grad_h
    
    # Map grad_h back to params (symmetric)
    # h[i] = params[i] for i < n/2
    # h[n-1-i] = params[i]
    # d(obj)/d(params[i]) = grad_h[i] + grad_h[n-1-i]
    grad_params = grad_h[:n_half] + grad_h[n-1:n_half-1:-1]
    
    return obj, grad_params

def _project_symmetric(params, target_sum):
    """Project symmetric params to satisfy sum(h) = target_sum and 0<=h<=1."""
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    # Target sum for h is n/2
    # Sum(h) = 2 * sum(params)
    # So we need sum(params) = n/4
    target_param_sum = n / 4.0
    
    # Project params to sum = target_param_sum, bounds [0,1]
    p = params.copy()
    # Simple projection: clip and adjust
    p = np.clip(p, 0, 1)
    diff = target_param_sum - np.sum(p)
    if abs(diff) > 1e-12:
        # Distribute diff to elements not at bounds
        # For simplicity, distribute evenly and clip
        p += diff / n_half
        p = np.clip(p, 0, 1)
        # Iterate a few times for better accuracy
        for _ in range(5):
            diff = target_param_sum - np.sum(p)
            if abs(diff) < 1e-12: break
            mask = (p > 0) & (p < 1)
            if np.sum(mask) == 0:
                # If all at bounds, we can't adjust. 
                # But target is n/4, max sum is n/2, min 0. 
                # n/4 is in range.
                break
            p[mask] += diff / np.sum(mask)
            p = np.clip(p, 0, 1)
    return p

def _brownian_bridge_path(n_half, seed=42):
    """Generate a Brownian bridge path scaled to fit [0,1] deviations."""
    np.random.seed(seed)
    # Brownian bridge: W(t) - t*W(1)
    # Discrete approximation
    increments = np.random.normal(0, 1, n_half)
    W = np.cumsum(increments)
    W_bridge = W - np.linspace(0, 1, n_half) * W[-1]
    # Scale to reasonable amplitude
    scale = 0.3
    return 0.5 + scale * W_bridge

def construct():
    n = N
    n_half = n // 2
    target_sum = n / 2.0
    target_param_sum = n / 4.0
    
    best_params = np.full(n_half, 0.5)
    best_obj = float('inf')
    
    # Calculate initial best
    h_init = np.concatenate([best_params, best_params[::-1]])
    corrs_init = _compute_overlaps(h_init)
    best_obj = 1.0 - np.min(corrs_init)
    
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n_half, 0.5))
    
    # Symmetric gradients
    restarts.append(np.linspace(0, 1, n_half))
    restarts.append(np.linspace(1, 0, n_half))
    
    # Symmetric sinusoids
    x = np.linspace(0, 2*np.pi, n_half)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    
    # Brownian bridge inspired restarts
    for seed in range(5):
        restarts.append(_brownian_bridge_path(n_half, seed=seed))
    
    # Random symmetric
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n_half))
    
    # Optimize
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        params = _project_symmetric(x0, target_param_sum)
        
        # Multi-stage beta optimization
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if _time_remaining() < 2.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_obj:
                    # Verify and project
                    params_new = _project_symmetric(res.x, target_param_sum)
                    h_new = np.concatenate([params_new, params_new[::-1]])
                    corrs_new = _compute_overlaps(h_new)
                    obj_new = 1.0 - np.min(corrs_new)
                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_params = params_new
                        params = params_new # Continue from here
            except Exception:
                pass
    
    # Final polish with full vector optimization if time permits
    if _time_remaining() > 10.0:
        h_full = np.concatenate([best_params, best_params[::-1]])
        
        def full_obj_and_grad(h):
            corrs = _compute_overlaps(h)
            min_val = np.min(corrs)
            shifted = corrs - min_val
            log_sum = np.log(np.sum(np.exp(-200.0 * shifted)))
            soft_min = min_val - log_sum / 200.0
            obj = 1.0 - soft_min
            
            exp_neg_beta = np.exp(-200.0 * corrs)
            sum_exp_neg = np.sum(exp_neg_beta)
            weights = exp_neg_beta / sum_exp_neg
            
            grad_h = np.zeros(n)
            for k in range(1, n):
                w = weights[k-1] * (2.0 / n)
                if w == 0: continue
                grad_h[:n-k] += w * h[k:]
                grad_h[k:] += w * h[:n-k]
            
            return obj, -grad_h

        try:
            res = minimize(
                full_obj_and_grad,
                h_full,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            h_new = np.clip(res.x, 0, 1)
            # Project to sum
            diff = target_sum - np.sum(h_new)
            if abs(diff) > 1e-12:
                mask = (h_new > 0) & (h_new < 1)
                if np.sum(mask) > 0:
                    h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
            
            corrs_new = _compute_overlaps(h_new)
            obj_new = 1.0 - np.min(corrs_new)
            if obj_new < best_obj:
                best_obj = obj_new
                # best_params not updated, but we return h_new
                best_params = None
                final_h = h_new
            else:
                final_h = np.concatenate([best_params, best_params[::-1]])
        except Exception:
            final_h = np.concatenate([best_params, best_params[::-1]])
    else:
        final_h = np.concatenate([best_params, best_params[::-1]])
    
    return final_h
