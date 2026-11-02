"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_correlations_fft(h):
    """Compute all correlations R(k) for k=1,...,n-1 using FFT.
    
    R(k) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    This is the autocorrelation of h, scaled.
    """
    n = len(h)
    # Use FFT for fast autocorrelation
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.rfft(h, size)
    autocorr = np.fft.irfft(H * np.conj(H), size)[:n]
    # autocorr[k] = sum_{i=0}^{n-k-1} h[i]*h[i+k]
    corrs = (2.0 / n) * autocorr[1:]
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation."""
    corrs = _compute_all_correlations_fft(h)
    return np.min(corrs)

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation."""
    h = x
    n = len(h)
    
    corrs = _compute_all_correlations_fft(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = min_val - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    weights = exp_neg_beta / sum_exp_neg
    
    # Compute gradient efficiently
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    w = np.clip(w, 0, 1)
    
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

def _local_search_subgradient(h, target_sum, time_budget):
    """Local subgradient search focusing on the minimum correlation shift."""
    n = len(h)
    best_h = h.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(best_h)
    step_size = 0.05
    no_improve = 0
    
    while _time_remaining() > time_budget:
        h = best_h
        corrs = _compute_all_correlations_fft(h)
        k_star = np.argmin(corrs) + 1  # 1-indexed
        min_corr = corrs[k_star - 1]
        
        grad = np.zeros(n)
        for j in range(n - k_star):
            grad[j] += (2.0 / n) * h[j + k_star]
            grad[j + k_star] += (2.0 / n) * h[j]
        
        h_new = h + step_size * grad
        h_new = _project_to_simplex(h_new, target_sum)
        
        new_obj = 1.0 - _compute_cross_correlation_min(h_new)
        if new_obj < best_obj - 1e-14:
            best_obj = new_obj
            best_h = h_new
            no_improve = 0
            step_size = min(step_size * 1.1, 0.1)
        else:
            no_improve += 1
            step_size *= 0.7
            if no_improve > 20:
                break
    
    return best_h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    np.random.seed(42)
    
    # Generate diverse restarts
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    # Block patterns
    for frac in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]:
        k = int(n * frac)
        val = 2.0 * target_sum / k if k > 0 else 0
        val = min(val, 1.0)
        k2 = n - k
        val2 = 2.0 * target_sum / k2 if k2 > 0 else 0
        val2 = min(val2, 1.0)
        restarts.append(np.concatenate([np.full(k, val), np.full(k2, 0.0)]))
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(k2, val2)]))
    
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Sinusoidal patterns
    x = np.linspace(0, 2 * np.pi, n)
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.2, 0.3, 0.4, 0.5]:
            restarts.append(np.clip(0.5 + amp * np.sin(freq * x), 0, 1))
            restarts.append(np.clip(0.5 + amp * np.cos(freq * x), 0, 1))
    
    # Random restarts
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Linear ramps
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    restarts.append(np.linspace(0.2, 0.8, n))
    restarts.append(np.linspace(0.8, 0.2, n))
    
    # Phase 1: Soft-min optimization with increasing beta
    beta_schedule = [5.0, 15.0, 50.0, 150.0, 500.0]
    
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: Local subgradient search on best solution
    if _time_remaining() > 20.0:
        try:
            h_local, obj_local = _local_search_subgradient(best_x, target_sum, 10.0)
            if obj_local < best_obj:
                best_obj = obj_local
                best_x = h_local
        except Exception:
            pass
    
    # Phase 3: High-beta polish with very tight tolerances
    if _time_remaining() > 15.0:
        for beta in [1000.0, 3000.0, 10000.0]:
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
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 4: Final local polish
    if _time_remaining() > 8.0:
        try:
            h_final, obj_final = _local_search_subgradient(best_x, target_sum, 4.0)
            if obj_final < best_obj:
                best_obj = obj_final
                best_x = h_final
        except Exception:
            pass
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
