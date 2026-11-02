"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_corrs(h):
    """Compute C_k for k=1..N-1. C_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _project_to_sum_bounds(h, target_sum):
    """Project h to satisfy sum(h)=target_sum and 0<=h<=1 using bisection."""
    # h is 1D array
    # We want to find offset 'c' such that sum(clip(h+c, 0, 1)) = target_sum
    # f(c) is monotonic increasing in c
    def sum_with_offset(c):
        return np.sum(np.clip(h + c, 0, 1))
    
    lo, hi = -1.0, 1.0
    # Ensure target_sum is within [sum(h<0), sum(h>1)]? 
    # Min sum is 0, max sum is N. target_sum is N/2.
    # So c will be in a reasonable range.
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = sum_with_offset(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    c = (lo + hi) / 2.0
    h_proj = np.clip(h + c, 0, 1)
    
    # Fine-tune to ensure exact sum
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-10:
        mask = (h_proj > 1e-12) & (h_proj < 1.0 - 1e-12)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0, 1)
            # If clipping changed the sum again, do one more pass
            diff2 = target_sum - np.sum(h_proj)
            if abs(diff2) > 1e-10:
                mask2 = (h_proj > 1e-12) & (h_proj < 1.0 - 1e-12)
                if np.sum(mask2) > 0:
                    h_proj[mask2] += diff2 / np.sum(mask2)
                    h_proj = np.clip(h_proj, 0, 1)
    return h_proj

def _soft_min_obj_and_grad(h, beta):
    """
    Objective: -soft_min(corrs)
    Gradient: -grad(soft_min)
    """
    n = len(h)
    corrs = _compute_corrs(h)
    
    # Soft min calculation
    min_val = np.min(corrs)
    shifted = corrs - min_val
    # To avoid overflow/underflow, we use shifted
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta)
    
    if sum_exp == 0:
        return 0.0, np.zeros(n)
        
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = -soft_min
    
    # Gradient of soft_min
    weights = exp_neg_beta / sum_exp
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w < 1e-12:
            continue
        # d(C_k)/d(h_i)
        # C_k = (2/n) sum_{i=0}^{n-k-1} h_i h_{i+k}
        # d/dh_i = (2/n) (h_{i+k} + h_{i-k}) (with boundary conditions)
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
        
    # Gradient of obj = -soft_min is -grad_h
    return obj, -grad_h

def construct():
    n = N
    target_sum = n / 2.0
    
    # Initializations
    inits = []
    
    # 1. Constant
    inits.append(np.full(n, 0.5))
    
    # 2. Low frequency cosine
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.clip(0.5 + 0.4 * np.cos(x), 0, 1))
    inits.append(np.clip(0.5 + 0.4 * np.sin(x), 0, 1))
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(0, 1, n))
    
    # 4. Block
    h_block = np.zeros(n)
    h_block[:n//2] = 1.0
    inits.append(h_block)
    
    best_h = inits[0].copy()
    best_score = -np.min(_compute_corrs(best_h)) # Maximize min corr
    
    # Project all inits
    for i in range(len(inits)):
        inits[i] = _project_to_sum_bounds(inits[i], target_sum)
        # Update best if this init is better
        score = np.min(_compute_corrs(inits[i]))
        if score > best_score:
            best_score = score
            best_h = inits[i].copy()

    # Optimization Loop
    for h_init in inits:
        if _time_remaining() < 15.0:
            break
            
        h_curr = h_init.copy()
        
        # Multi-stage beta optimization
        for beta in [20.0, 100.0, 500.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    h_curr,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                h_new = np.clip(res.x, 0, 1)
                h_new = _project_to_sum_bounds(h_new, target_sum)
                
                corrs = _compute_corrs(h_new)
                min_corr = np.min(corrs)
                
                if min_corr > best_score:
                    best_score = min_corr
                    best_h = h_new.copy()
                    h_curr = h_new.copy()
            except Exception:
                pass
        
        # Subgradient Polish
        if _time_remaining() < 5.0:
            continue
            
        h_curr = best_h.copy()
        lr = 0.05
        for step in range(1000):
            if _time_remaining() < 2.0:
                break
                
            corrs = _compute_corrs(h_curr)
            min_idx = np.argmin(corrs)
            k = min_idx + 1
            
            # Gradient of C_k
            grad = np.zeros(n)
            grad[:n-k] += h_curr[k:]
            grad[k:] += h_curr[:n-k]
            grad *= (2.0 / n)
            
            # Update
            h_curr += lr * grad
            h_curr = _project_to_sum_bounds(h_curr, target_sum)
            
            # Check improvement
            new_corr = _compute_corrs(h_curr)
            new_min = np.min(new_corr)
            
            if new_min > best_score:
                best_score = new_min
                best_h = h_curr.copy()
                # Increase lr slightly if improving? Or keep.
                # If not improving, decrease lr
            else:
                # If no improvement, reduce learning rate
                lr *= 0.95
                if lr < 1e-6:
                    break

    # Final Projection and Return
    best_h = _project_to_sum_bounds(best_h, target_sum)
    return best_h
# EVOLVE-BLOCK-END
