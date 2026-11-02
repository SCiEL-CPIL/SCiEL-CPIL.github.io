"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, brentq

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
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * (h[j+k] if j+k < n else 0) + (2/n) * (h[j-k] if j-k >= 0 else 0)
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w != 0:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h) = -grad
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # We want to find theta such that sum(clip(v - theta, 0, 1)) = target_sum
    # Let f(theta) = sum(clip(v - theta, 0, 1)) - target_sum
    # f is continuous and monotonically decreasing.
    
    def f(theta):
        w = np.clip(v - theta, 0, 1)
        return np.sum(w) - target_sum
    
    # Find bounds for brentq
    lo = np.min(v) - 1.0
    hi = np.max(v) + 1.0
    
    # Ensure signs are opposite for brentq
    # If f(lo) < 0, we need smaller theta (which increases sum)
    # If f(hi) > 0, we need larger theta (which decreases sum)
    # But f is decreasing. So f(lo) >= f(hi).
    # We want f(theta) = 0.
    # If f(lo) < 0, then for all theta >= lo, f(theta) <= f(lo) < 0.
    # This implies sum(clip(v - theta, 0, 1)) < target_sum for all theta >= lo.
    # Max sum is at theta = -inf (conceptually), which is sum(v) if v > 0.
    # Actually, as theta -> -inf, clip(v - theta, 0, 1) -> 1 for all v. Sum -> n.
    # So f(-inf) = n - target_sum > 0 (since target_sum = n/2).
    # So there must be a root.
    # We just need to expand lo if f(lo) < 0.
    if f(lo) < 0:
        lo -= 10.0
    if f(hi) > 0:
        hi += 10.0
        
    try:
        theta = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        # Fallback to iterative
        w = np.clip(v, 0, 1)
        for _ in range(50):
            diff = target_sum - np.sum(w)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = w < 1.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            else:
                mask = w > 0.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
        return w

    w = np.clip(v - theta, 0, 1)
    # Correct for any tiny numerical error in sum
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-12:
        # Distribute diff to elements not at bounds
        mask = (w > 1e-9) & (w < 1.0 - 1e-9)
        cnt = np.sum(mask)
        if cnt > 0:
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
    # More diverse sinusoidal mixes
    np.random.seed(123)
    for _ in range(5):
        freq = np.random.randint(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.2, 0.4)
        restarts.append(0.5 + amp*np.sin(freq*x + phase))
    
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
        for beta in [500.0, 1000.0, 2000.0]:
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
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
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
        for _ in range(200):
            if _time_remaining() < 5.0:
                break
            h = best_x
            n = len(h)
            
            # Use FFT to find min corr
            corrs = _compute_all_corrs_fft(h)
            min_idx = np.argmin(corrs)
            min_corr = corrs[min_idx]
            k_star = min_idx + 1
            
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            # Line search along grad
            best_t = 0.0
            current_obj = 1.0 - min_corr
            t = 0.1
            for _ in range(10):
                h_trial = h + t * grad
                h_trial = _project_to_simplex(h_trial, target_sum)
                trial_corr = _compute_cross_correlation_min(h_trial)
                trial_obj = 1.0 - trial_corr
                if trial_obj < current_obj:
                    current_obj = trial_obj
                    best_t = t
                    best_x = h_trial
                t *= 0.5
                
            if best_t == 0.0:
                break
    
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
