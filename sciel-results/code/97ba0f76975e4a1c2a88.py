"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_min_corr(h):
    """Compute min over shifts of the cross-correlation."""
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _compute_all_corrs(h):
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _soft_min_obj_grad(x, beta=100.0):
    h = x
    n = len(h)
    corrs = _compute_all_corrs(h)
    
    # stable log-sum-exp
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    obj = 1.0 - soft_min
    
    # weights
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp
    
    # gradient of soft_min wrt h
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    grad_obj = -grad
    return obj, grad_obj

def _project(v, target_sum):
    """Project onto {x: 0<=x<=1, sum=target_sum}."""
    n = len(v)
    # simplex projection first (x>=0, sum=target)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_cands = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_cands) == 0:
        rho = n - 1
    else:
        rho = rho_cands[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    # iterate to fix sum with bounds
    for _ in range(50):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
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

def _symmetric_params_to_h(params):
    """Convert symmetric params (length N/2) to full h (length N)."""
    n_half = len(params)
    h = np.zeros(2 * n_half)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    return h

def _h_to_symmetric_params(h):
    """Extract symmetric params from full h (assuming h is symmetric)."""
    n_half = len(h) // 2
    # Average the two halves to get best symmetric approximation
    params = 0.5 * (h[:n_half] + h[n_half-1::-1])
    return params

def _symmetric_obj_grad(params, beta=100.0):
    """Objective and gradient for symmetric h."""
    n_half = len(params)
    n = 2 * n_half
    h = _symmetric_params_to_h(params)
    
    corrs = _compute_all_corrs(h)
    
    # Soft min
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    obj = 1.0 - soft_min
    
    # Weights
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp
    
    # Gradient wrt h
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad_h[:n-k] += w * h[k:]
            grad_h[k:] += w * h[:n-k]
    
    grad_h = -grad_h
    
    # Map to symmetric params
    # h[i] = params[i] for i < n_half
    # h[n-1-i] = params[i]
    # d(obj)/d(params[i]) = grad_h[i] + grad_h[n-1-i]
    grad_params = grad_h[:n_half] + grad_h[n-1:n_half-1:-1]
    
    return obj, grad_params

def _project_symmetric(params, target_param_sum):
    """Project symmetric params to satisfy sum(params) = target_param_sum and 0<=params<=1."""
    n_half = len(params)
    p = params.copy()
    p = np.clip(p, 0, 1)
    
    # Simple iterative projection
    for _ in range(100):
        diff = target_param_sum - np.sum(p)
        if abs(diff) < 1e-14:
            break
        mask = (p > 0) & (p < 1)
        cnt = np.sum(mask)
        if cnt == 0:
            break
        p[mask] += diff / cnt
        p = np.clip(p, 0, 1)
    
    return p

def construct():
    n = N
    n_half = n // 2
    target_sum = n / 2.0
    target_param_sum = n / 4.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_x)
    
    np.random.seed(42)
    
    # Generate restarts for symmetric params
    restarts_params = []
    restarts_params.append(np.full(n_half, 0.5))
    
    # Symmetric gradients
    restarts_params.append(np.linspace(0, 1, n_half))
    restarts_params.append(np.linspace(1, 0, n_half))
    
    # Symmetric sinusoids
    x = np.linspace(0, 2*np.pi, n_half)
    for freq, amp in [(1, 0.3), (2, 0.3), (3, 0.2), (1, 0.4), (2, 0.4), (4, 0.2)]:
        restarts_params.append(0.5 + amp*np.sin(freq*x))
        restarts_params.append(0.5 + amp*np.cos(freq*x))
    
    # Random symmetric
    for _ in range(20):
        restarts_params.append(np.random.uniform(0, 1, n_half))
    
    # Random smooth
    for _ in range(10):
        coeffs = np.random.randn(1, 5, 3) * 0.1
        t = np.linspace(0, 2*np.pi, n_half)
        sig = 0.5
        for j in range(1, 5):
            for k in range(1, 3):
                sig += coeffs[0, j, k] * np.sin(j * k * t)
        restarts_params.append(np.clip(sig, 0, 1))
    
    # Block patterns
    for k in range(1, n_half):
        restarts_params.append(np.concatenate([np.zeros(k), np.ones(n_half-k)*0.5]))
        restarts_params.append(np.concatenate([np.ones(k)*0.5, np.zeros(n_half-k)]))
    
    # Alternating patterns
    alt1 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n_half)])
    alt2 = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n_half)])
    restarts_params.append(alt1)
    restarts_params.append(alt2)
    
    beta_schedule = [15.0, 40.0, 100.0, 250.0, 500.0]
    
    # Phase 1: Optimize in symmetric parameter space
    for x0 in restarts_params:
        if _time_remaining() < 8.0:
            break
        params = _project_symmetric(x0, target_param_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _symmetric_obj_grad, params, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 30}
                )
                params_new = _project_symmetric(res.x, target_param_sum)
                h_new = _symmetric_params_to_h(params_new)
                h_new = _project(h_new, target_sum)
                # Re-extract symmetric params after projection
                params_new = _h_to_symmetric_params(h_new)
                robj = 1.0 - _compute_min_corr(h_new)
                if robj < best_obj:
                    best_obj = robj
                    best_x = h_new
                params = params_new
            except Exception:
                pass
    
    # Phase 2: High-beta polish in symmetric space
    if _time_remaining() > 15.0:
        current_params = _h_to_symmetric_params(best_x)
        for beta in [800.0, 1500.0, 3000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _symmetric_obj_grad, current_params, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 20}
                )
                params_new = _project_symmetric(res.x, target_param_sum)
                h_new = _symmetric_params_to_h(params_new)
                h_new = _project(h_new, target_sum)
                params_new = _h_to_symmetric_params(h_new)
                robj = 1.0 - _compute_min_corr(h_new)
                if robj < best_obj:
                    best_obj = robj
                    best_x = h_new
                current_params = params_new
            except Exception:
                pass
    
    # Phase 3: Full vector optimization (break symmetry if beneficial)
    if _time_remaining() > 10.0:
        for beta in [100.0, 300.0, 1000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad, best_x, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 25}
                )
                h_new = _project(res.x, target_sum)
                robj = 1.0 - _compute_min_corr(h_new)
                if robj < best_obj:
                    best_obj = robj
                    best_x = h_new
            except Exception:
                pass
    
    # Phase 4: Subgradient ascent on active shift
    if _time_remaining() > 10.0:
        step_size = 0.005
        for _ in range(300):
            if _time_remaining() < 5.0:
                break
            h = best_x
            n = len(h)
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                c = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if c < min_corr:
                    min_corr = c
                    k_star = k
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            h_new = h + step_size * grad
            h_new = _project(h_new, target_sum)
            new_obj = 1.0 - _compute_min_corr(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
            else:
                step_size *= 0.7
    
    # Exact sum fix
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-13:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
