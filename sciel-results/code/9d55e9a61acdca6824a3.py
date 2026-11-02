"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs(h):
    """Compute all cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation."""
    return np.min(_compute_all_corrs(h))

def _soft_min_and_grad(h, beta=100.0):
    """Compute soft-min of correlations and its gradient w.r.t. h.
    
    soft_min = -(1/beta) * log(sum_k exp(-beta * R_k))
    grad = sum_k w_k * dR_k/dh where w_k = exp(-beta*R_k) / sum_j exp(-beta*R_j)
    """
    n = len(h)
    corrs = _compute_all_corrs(h)
    
    # Numerically stable log-sum-exp
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights for gradient
    weights = exp_terms / sum_exp  # w_k = exp(-beta*(R_k - min_val)) / sum_j exp(-beta*(R_j - min_val))
    
    # Gradient: d(soft_min)/d(h) = sum_k w_k * d(R_k)/d(h)
    # R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(R_k)/d(h[j]) = (2/n)*h[j+k] for j < n-k, plus (2/n)*h[j-k] for j >= k
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    return soft_min, grad

def _objective_and_grad_soft(x, beta=100.0):
    """Objective: minimize 1 - soft_min_corr, and its gradient."""
    soft_min, grad = _soft_min_and_grad(x, beta)
    obj = 1.0 - soft_min
    grad_obj = -grad
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
    for _ in range(20):  # Iterate a few times
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-13:
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

def _fista_optimize(x0, target_sum, beta, max_iter=500, tol=1e-10):
    """FISTA optimization with adaptive step size on the smooth surrogate."""
    n = len(x0)
    x = x0.copy()
    y = x.copy()
    t = 1.0
    L = 1.0  # Lipschitz constant estimate
    
    best_obj = 1.0 - _compute_cross_correlation_min(x)
    
    for it in range(max_iter):
        if _time_remaining() < 3.0:
            break
        
        # Compute gradient at y
        obj_y, grad_y = _objective_and_grad_soft(y, beta)
        
        # Gradient step
        z = y - (1.0 / L) * grad_y
        z = _project_to_simplex(z, target_sum)
        
        # Line search: find step size such that f(z) <= f(y) + <grad_y, z-y> + L/2*||z-y||^2
        step = 1.0
        while step > 1e-8:
            obj_z, _ = _objective_and_grad_soft(z, beta)
            if obj_z <= obj_y + np.dot(grad_y, z - y) + 0.5 * L * step * np.dot(z - y, z - y):
                break
            step *= 0.5
            z = y - (step / L) * grad_y
            z = _project_to_simplex(z, target_sum)
        
        # FISTA update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = z + ((t - 1.0) / t_new) * (z - x)
        y = _project_to_simplex(y, target_sum)
        x = z
        t = t_new
        L = L / step  # Update Lipschitz estimate
        
        # Check convergence
        obj_x = 1.0 - _compute_cross_correlation_min(x)
        if obj_x < best_obj:
            best_obj = obj_x
        if abs(obj_y - (1.0 - _compute_cross_correlation_min(x))) < tol:
            break
    
    return x, best_obj

def _local_search_max_overlap(x, target_sum, max_iter=200):
    """Local search that directly optimizes the maximum overlap using subgradient."""
    n = len(x)
    x_best = x.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(x_best)
    step_size = 0.05
    
    for _ in range(max_iter):
        if _time_remaining() < 2.0:
            break
        
        corrs = _compute_all_corrs(x_best)
        min_idx = np.argmin(corrs)
        k_star = min_idx + 1
        
        # Subgradient of -min_corr = max_overlap
        grad = np.zeros(n)
        for j in range(n - k_star):
            grad[j] += (2.0 / n) * x_best[j + k_star]
            grad[j + k_star] += (2.0 / n) * x_best[j]
        
        # We want to maximize min_corr, so move in +grad direction
        x_new = x_best + step_size * grad
        x_new = _project_to_simplex(x_new, target_sum)
        
        new_obj = 1.0 - _compute_cross_correlation_min(x_new)
        if new_obj < best_obj:
            best_obj = new_obj
            x_best = x_new
            step_size = min(step_size * 1.1, 0.5)
        else:
            step_size *= 0.7
    
    return x_best, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
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
    
    # Optimize each restart with FISTA on soft-min objective
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # Use FISTA with moderate beta for smoothness
        try:
            res_x, res_obj = _fista_optimize(x, target_sum, beta=50.0, max_iter=300)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 10.0:
        for beta in [100.0, 200.0, 400.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res_x, res_obj = _fista_optimize(best_x, target_sum, beta=beta, max_iter=400)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final polish with local search on hard max overlap
    if _time_remaining() > 8.0:
        try:
            res_x, res_obj = _local_search_max_overlap(best_x, target_sum, max_iter=300)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
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
