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

def _compute_all_corrs_fft(h):
    """Compute all cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1
    using FFT for O(n log n) complexity."""
    n = len(h)
    # Zero-pad to avoid circular correlation artifacts
    size = 2 * n
    f = np.fft.rfft(h, size)
    # Cross-correlation via FFT: corr[k] = sum_i h[i]*h[i+k]
    # This is the autocorrelation shifted
    corr_full = np.fft.irfft(f * np.conj(f), size)
    # corr_full[k] = sum_i h[i]*h[i+k] for k=0..n-1 (with proper indexing)
    # Actually, for autocorrelation: corr_full[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k]
    corrs = corr_full[1:n] * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

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
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0 - 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    # Final correction for any tiny residual
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-15:
        mask = (w > 1e-15) & (w < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            w[mask] += diff / np.sum(mask)
            w = np.clip(w, 0, 1)
    
    return w

def _soft_min_and_grad(h, beta):
    """Compute soft_min of correlations and its gradient w.r.t. h, using FFT."""
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    weights = exp_terms / sum_exp  # shape (n-1,)
    
    # Vectorized gradient: for each j, grad[j] = (2/n) * sum_{k=1}^{n-1} w_k * (h[j+k] + h[j-k] where valid)
    # This is a weighted autocorrelation of h with itself, weighted by weights
    # grad[j] = (2/n) * sum_{k} weights[k-1] * h[j+k] for j < n-k
    #          + (2/n) * sum_{k} weights[k-1] * h[j-k] for j >= k
    
    # We can compute this via convolution: grad = (2/n) * (h * w_rev + w_rev * h) where w_rev is reversed weights
    # Actually, let's use a direct approach with numpy broadcasting for moderate n
    # For n=100, the loop is only 99 iterations which is fine
    
    grad = np.zeros(n)
    # Pre-filter weights that are negligible
    active = weights > 1e-15
    if np.any(active):
        for k_idx in np.where(active)[0]:
            k = k_idx + 1
            w = weights[k_idx] * (2.0 / n)
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    return soft_min, grad

def _epigraph_objective_and_grad(x, beta):
    """Epigraph formulation: minimize t subject to t >= 1 - R_k for all k.
    
    We use a smooth approximation: minimize t + (1/beta)*log(sum_k exp(beta*(1-R_k - t)))
    This is the soft-max of (1-R_k - t) plus t.
    
    Variables: x = [h_0, ..., h_{n-1}, t]
    """
    n = len(x) - 1
    h = x[:n]
    t = x[n]
    
    corrs = _compute_all_corrs_fft(h)
    
    # We want to minimize max_k (1 - R_k), which is 1 - min_k R_k
    # Soft max: (1/beta) * log(sum_k exp(beta * (1 - R_k)))
    # But we also have the epigraph variable t
    # Objective: t + (1/beta) * log(sum_k exp(beta * (1 - R_k - t)))
    
    vals = 1.0 - corrs - t  # shape (n-1,)
    max_val = np.max(vals)
    shifted = vals - max_val
    exp_terms = np.exp(beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_max = max_val + log_sum / beta
    
    obj = t + soft_max
    
    # Gradient w.r.t. t:
    # d(obj)/dt = 1 + d(soft_max)/dt = 1 - (sum_k exp(beta*(1-R_k-t)) / sum_j exp(beta*(1-R_j-t)))
    #           = 1 - 1 = 0? No wait...
    # Actually: d(soft_max)/dt = (1/beta) * d/dt log(sum_k exp(beta*(1-R_k-t)))
    #           = (1/beta) * sum_k exp(beta*(1-R_k-t)) * (-beta) / sum_j exp(beta*(1-R_j-t))
    #           = -1
    # So d(obj)/dt = 1 - 1 = 0? That can't be right for optimization.
    
    # Let me reconsider. The epigraph formulation is:
    # minimize t
    # subject to: 1 - R_k <= t for all k
    # Smoothed: minimize t + (1/beta) * sum_k max(0, 1-R_k-t)^2 or use log-sum-exp
    
    # Actually, let's use: minimize t + (1/beta) * log(sum_k exp(beta*(1-R_k-t)))
    # This is a barrier/penalty formulation.
    
    # d(soft_max)/dt where soft_max = (1/beta)*log(sum_k exp(beta*(1-R_k-t)))
    # = (1/beta) * [sum_k exp(beta*(1-R_k-t)) * (-beta)] / [sum_j exp(beta*(1-R_j-t))]
    # = -1
    
    # So d(obj)/dt = 1 + (-1) = 0. That means t is not determined by the gradient.
    # This is expected in epigraph - we need to handle t differently.
    
    # Let's use a different approach: directly minimize the soft-max of (1-R_k)
    # which is 1 - soft_min(R_k)
    
    # Revert to the soft-min approach but with FFT acceleration
    # The key improvement is the FFT-based correlation computation
    
    # For now, let's just return the soft-min objective and gradient
    soft_min, grad_h = _soft_min_and_grad(h, beta)
    obj = 1.0 - soft_min
    grad_full = np.zeros(n + 1)
    grad_full[:n] = -grad_h
    # t is not used in this formulation, set its gradient to 0
    grad_full[n] = 0.0
    
    return obj, grad_full

def _soft_min_obj_grad(h, beta):
    """Objective and gradient for soft-min formulation (no epigraph)."""
    soft_min, grad_h = _soft_min_and_grad(h, beta)
    obj = 1.0 - soft_min
    grad = -grad_h
    return obj, grad

def _coordinate_polish(h, target_sum, max_iters=50):
    """Efficient coordinate-wise descent to polish the solution."""
    n = len(h)
    h = h.copy()
    h = _project_to_simplex(h, target_sum)
    
    best_obj = 1.0 - _compute_cross_correlation_min(h)
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for i in range(n):
            if _time_remaining() < 5.0:
                break
            
            old_val = h[i]
            found_improvement = False
            for delta in step_sizes:
                for sign in [1, -1]:
                    new_val = old_val + sign * delta
                    if new_val < 0 or new_val > 1:
                        continue
                    
                    # Create candidate: change h[i], distribute sum diff to others
                    h[i] = new_val
                    diff = target_sum - np.sum(h)
                    if abs(diff) > 1e-15:
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        cnt = np.sum(mask)
                        if cnt == 0:
                            h[i] = old_val
                            continue
                        h[mask] += diff / cnt
                        h = np.clip(h, 0, 1)
                        h = _project_to_simplex(h, target_sum)
                    
                    new_obj = 1.0 - _compute_cross_correlation_min(h)
                    
                    if new_obj < best_obj - 1e-13:
                        best_obj = new_obj
                        improved = True
                        found_improvement = True
                        break
                    else:
                        h[i] = old_val
                if found_improvement:
                    break
            
            if not improved and it > 20:
                break
    
    return h, best_obj

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
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if k > 0 and n - k > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # More bang-bang patterns
    for period in [3, 4, 5, 7]:
        pattern = [0.0 if (i // period) % 2 == 0 else 1.0 for i in range(n)]
        arr = np.array(pattern)
        # Scale to have correct sum
        curr_sum = np.sum(arr)
        if curr_sum > 0:
            arr = arr * (target_sum / curr_sum)
            arr = np.clip(arr, 0, 1)
            arr = _project_to_simplex(arr, target_sum)
            restarts.append(arr)
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    # Near-optimal structured patterns: try patterns with low autocorrelation
    # A good candidate: h[i] = 0.5 + a*cos(2*pi*i/p) for various periods
    for p in [3, 4, 5, 6, 7, 8]:
        for a in [0.1, 0.2, 0.3, 0.4]:
            pattern = 0.5 + a * np.cos(2 * np.pi * np.arange(n) / p)
            pattern = np.clip(pattern, 0, 1)
            pattern = _project_to_simplex(pattern, target_sum)
            restarts.append(pattern)
            pattern2 = 0.5 + a * np.sin(2 * np.pi * np.arange(n) / p)
            pattern2 = np.clip(pattern2, 0, 1)
            pattern2 = _project_to_simplex(pattern2, target_sum)
            restarts.append(pattern2)
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for amp in [0.2, 0.3, 0.4, 0.5]:
        for freq in [1, 2, 3]:
            restarts.append(0.5 + amp*np.sin(freq*x))
            restarts.append(0.5 + amp*np.cos(freq*x))
    
    # Optimize each restart with soft-min objective
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # Use soft-min with moderate beta for smoothness
        try:
            res = minimize(
                _soft_min_obj_grad,
                x,
                args=(20.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = 1.0 - _compute_cross_correlation_min(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 20.0:
        for beta in [40.0, 80.0, 150.0, 300.0]:
            if _time_remaining() < 10.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final coordinate-wise polish
    if _time_remaining() > 8.0:
        best_x, best_obj = _coordinate_polish(best_x, target_sum, max_iters=60)
    
    # One more L-BFGS-B pass with very high beta if time allows
    if _time_remaining() > 12.0:
        try:
            res = minimize(
                _soft_min_obj_grad,
                best_x,
                args=(500.0,),
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
        except Exception:
            pass
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-14:
        mask = (best_x > 1e-15) & (best_x < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
