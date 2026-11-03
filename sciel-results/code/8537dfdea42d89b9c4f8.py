"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s).
    
    We evaluate at discrete shifts k = 1, ..., n-1.
    R(k) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    The overlap O(k) = 1 - R(k) (since integral of h is 1, integral of 1-h is 1, and integral of h*(1-h_shifted) = 1 - R(k)).
    Wait, let's re-verify the overlap definition.
    Task: minimize maximum overlap of h with shifted 1-h.
    Overlap(k) = ∫ h(x) * (1 - h(x+k)) dx = ∫ h(x) dx - ∫ h(x)h(x+k) dx = 1 - R(k).
    So minimizing max overlap is equivalent to maximizing min R(k).
    """
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _max_overlap(h):
    """Compute the maximum overlap value."""
    return 1.0 - _compute_cross_correlation_min(h)

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(k) using log-sum-exp.
    
    soft_min = -(1/beta) * log(sum_k exp(-beta * R_k))
    """
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation.
    
    We want to minimize max overlap = 1 - min_corr.
    So we minimize (1 - soft_min_corr).
    """
    h = x
    n = len(h)
    
    # Compute all correlations
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Soft min
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient of soft_min w.r.t. h
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg  # softmax weights
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * h[j+k] for j < n-k
    #                    + (2/n) * h[j-k] for j >= k
    
    grad_soft_min = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w != 0:
            grad_soft_min[:n-k] += w * h[k:]
            grad_soft_min[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h)
    grad_obj = -grad_soft_min
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(10):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        if k > 0 and k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    
    # Optimize each restart with soft-min objective
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # Use soft-min with moderate beta for smoothness
        try:
            res = minimize(
                _objective_and_grad_soft,
                x,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = _max_overlap(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = _max_overlap(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final polish with hard min using subgradient descent on active constraints
    if _time_remaining() > 5.0:
        h = best_x.copy()
        lr = 0.001
        n_iter = 5000
        for it in range(n_iter):
            if _time_remaining() < 1.0:
                break
            min_corr = _compute_cross_correlation_min(h)
            # Find the k that gives the minimum correlation
            corrs = []
            for k in range(1, n):
                corrs.append(np.dot(h[:n-k], h[k:]) * (2.0 / n))
            corrs = np.array(corrs)
            min_idx = np.argmin(corrs) + 1  # k value
            
            # Subgradient of min correlation
            # d(R_k)/d(h[j]) = (2/n) * h[j+k] for j < n-k
            #                + (2/n) * h[j-k] for j >= k
            grad = np.zeros(n)
            k = min_idx
            grad[:n-k] += (2.0/n) * h[k:]
            grad[k:] += (2.0/n) * h[:n-k]
            
            # Also consider nearby k values for robustness
            for dk in [-2, -1, 1, 2]:
                k2 = min_idx + dk
                if 1 <= k2 < n:
                    gap = corrs[min_idx-1] - corrs[k2-1]
                    if gap < 0.01:
                        grad[:n-k2] += 0.5 * (2.0/n) * h[k2:]
                        grad[k2:] += 0.5 * (2.0/n) * h[:n-k2]
            
            # We want to maximize min correlation, so we do gradient ascent
            h_new = h + lr * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            cur_obj = _max_overlap(h_new)
            if cur_obj < best_obj - 1e-12:
                best_obj = cur_obj
                best_x = h_new.copy()
            
            h = h_new
    
    # Ensure exact sum
    best_x = _project_to_simplex(best_x, target_sum)
    
    return best_x
# EVOLVE-BLOCK-END
