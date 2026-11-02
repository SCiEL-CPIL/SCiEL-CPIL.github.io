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
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    # Zero-pad to avoid circular correlation
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    # Autocorrelation via FFT: ifft(|H|^2) gives circular autocorrelation
    # For linear autocorrelation, we need to be careful
    # Alternative: compute directly but vectorized
    # R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # This is the linear autocorrelation
    # Use FFT: pad h to length 2n, compute autocorrelation
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    # acf[k] = sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=0,...,n-1
    # (the linear autocorrelation)
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp."""
    corrs = _compute_all_corrs_fft(h)
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based correlations."""
    h = x
    n = len(h)
    
    # Compute all correlations via FFT
    corrs = _compute_all_corrs_fft(h)
    
    # Clean soft-min formulation: soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = -np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    weights = exp_neg_beta / sum_exp_neg
    w_k = weights * (2.0 / n)
    
    # Vectorized gradient computation
    # d(soft_min)/d(h[j]) = sum_{k=1}^{n-1} w_k * (h[j+k] + h[j-k])
    
    if n > 1:
        j_idx = np.arange(n)
        k_idx = np.arange(1, n)
        J, K = np.meshgrid(j_idx, k_idx, indexing='ij')
        
        # h[j+k]
        mask_r = (J + K) < n
        h_r = np.zeros((n, n-1))
        h_r[mask_r] = h[J[mask_r] + K[mask_r]]
        
        # h[j-k]
        mask_l = (J - K) >= 0
        h_l = np.zeros((n, n-1))
        h_l[mask_l] = h[J[mask_l] - K[mask_l]]
        
        grad_soft_min = (h_r + h_l) @ w_k
        grad_obj = -grad_soft_min
    else:
        grad_obj = np.zeros(n)
        
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
    for _ in range(10):  # Iterate a few times
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
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
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
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: High-beta polish on best solution
    if _time_remaining() > 15.0:
        for beta in [500.0, 1000.0, 2000.0, 5000.0]:
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
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 3: Local search with subgradient on the active constraint
    if _time_remaining() > 10.0:
        step_size = 0.01
        for _ in range(200):
            if _time_remaining() < 5.0:
                break
            h = best_x
            n = len(h)
            
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if corr < min_corr:
                    min_corr = corr
                    k_star = k
            
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
            else:
                step_size *= 0.5
    
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
