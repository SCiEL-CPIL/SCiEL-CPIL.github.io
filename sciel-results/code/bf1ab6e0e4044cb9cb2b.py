"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _all_corrs_fft(h):
    """Compute all circular cross-correlations R(k) = sum_i h[i]*h[(i+k)%n] using FFT.
    Returns array of length n where corrs[0] = sum h^2 (DC), corrs[k] for k=1..n-1.
    """
    n = len(h)
    H = np.fft.rfft(h, n=2*n)
    corr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    return corr

def _linear_corrs(h):
    """Compute linear (non-circular) correlations R(k) = sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:])
    return corrs

def _compute_min_corr(h):
    """Compute min over shifts of the linear cross-correlation R(s)."""
    n = len(h)
    corrs = _linear_corrs(h)
    return np.min(corrs) * (2.0 / n)

def _max_overlap(h):
    """Compute the maximum overlap (objective to minimize)."""
    return 1.0 - _compute_min_corr(h)

def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    v = np.clip(v, 0, 1)
    # Iterative adjustment
    for _ in range(50):
        diff = target_sum - np.sum(v)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = v < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            v[mask] += diff / cnt
        else:
            mask = v > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            v[mask] += diff / cnt
        v = np.clip(v, 0, 1)
    return v

def _soft_min_obj_grad(h, beta):
    """Soft-min objective and gradient using FFT-accelerated correlations.
    Returns (obj, grad) where obj = 1 - soft_min_corr.
    """
    n = len(h)
    # Use circular correlations as a proxy (faster via FFT)
    corr_circ = _all_corrs_fft(h)
    # Extract k=1..n-1
    corrs = corr_circ[1:]
    corrs = corrs * (2.0 / n)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta)
    soft_min = min_val - np.log(sum_exp) / beta
    obj = 1.0 - soft_min
    
    # Gradient: d(soft_min)/d(h) = sum_k w_k * d(R_k)/d(h)
    # For circular correlation, d(R_k)/d(h_i) = h[(i+k)%n] + h[(i-k)%n]
    weights = exp_neg_beta / sum_exp
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad += w * (np.roll(h, k) + np.roll(h, -k))
    
    grad_obj = -grad
    return obj, grad_obj

def _subgradient_ascent(h, target_sum, n_iters, lr_init=0.02):
    """Subgradient ascent on min-correlation (equivalent to descent on max-overlap).
    Uses active-set tracking for the minimum correlation shift.
    """
    n = len(h)
    best_h = h.copy()
    best_obj = _max_overlap(best_h)
    lr = lr_init
    
    for it in range(n_iters):
        if _time_remaining() < 3.0:
            break
        
        # Find active set: shifts where correlation is near minimum
        corrs = _linear_corrs(h) * (2.0 / n)
        min_corr = np.min(corrs)
        tol = 0.05
        active = np.where(corrs < min_corr + tol)[0] + 1  # k values
        
        # Compute subgradient: average of gradients over active set
        grad = np.zeros(n)
        for k in active:
            for j in range(n - k):
                grad[j] += (2.0 / n) * h[j + k]
                grad[j + k] += (2.0 / n) * h[j]
        grad /= len(active)
        
        # Take step
        h_new = h + lr * grad
        h_new = _project(h_new, target_sum)
        
        new_obj = _max_overlap(h_new)
        if new_obj < best_obj:
            best_obj = new_obj
            best_h = h_new.copy()
            h = h_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
    
    return best_h, best_obj

def _coordinate_descent(h, target_sum, n_iters, lr_init=0.01):
    """Coordinate descent on max-overlap objective."""
    n = len(h)
    best_h = h.copy()
    best_obj = _max_overlap(best_h)
    lr = lr_init
    
    for it in range(n_iters):
        if _time_remaining() < 3.0:
            break
        
        improved = False
        for i in range(n):
            if _time_remaining() < 3.0:
                break
            
            # Try increasing and decreasing h[i]
            for delta in [lr, -lr]:
                h_new = h.copy()
                h_new[i] += delta
                h_new = _project(h_new, target_sum)
                
                new_obj = _max_overlap(h_new)
                if new_obj < best_obj - 1e-15:
                    best_obj = new_obj
                    best_h = h_new.copy()
                    h = h_new
                    improved = True
                    break
        
        if not improved:
            lr *= 0.8
            if lr < 1e-7:
                break
    
    return best_h, best_obj

def _anneal_subgradient(h, target_sum, n_total_iters):
    """Simulated annealing with subgradient steps."""
    n = len(h)
    best_h = h.copy()
    best_obj = _max_overlap(best_h)
    temp = 0.1
    cooling = 0.995
    
    for it in range(n_total_iters):
        if _time_remaining() < 3.0:
            break
        
        # Random perturbation
        mask = np.random.random(n) < 0.1
        perturbation = np.random.uniform(-0.1, 0.1, n) * mask
        h_pert = _project(h + perturbation, target_sum)
        
        obj_pert = _max_overlap(h_pert)
        
        # Accept if better or with probability exp(-delta/temp)
        delta = obj_pert - _max_overlap(h)
        if delta < 0 or np.random.random() < np.exp(-delta / (temp + 1e-10)):
            h = h_pert
            if obj_pert < best_obj:
                best_obj = obj_pert
                best_h = h.copy()
        
        temp *= cooling
    
    return best_h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if k > 0 and k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, target_sum/k), np.full(n-k, 0.0)]))
    
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 10]:
        pattern = np.array([1.0 if (i % period) < period//2 else 0.0 for i in range(n)])
        restarts.append(pattern)
        restarts.append(1.0 - pattern)
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.3, 0.4, 0.5]:
            restarts.append(0.5 + amp*np.sin(freq*x))
            restarts.append(0.5 + amp*np.cos(freq*x))
    
    # Random
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Phase 1: Subgradient ascent from multiple restarts
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        x0 = _project(x0, target_sum)
        h_opt, obj_opt = _subgradient_ascent(x0, target_sum, n_iters=500, lr_init=0.02)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = h_opt
    
    # Phase 2: Coordinate descent polish
    if _time_remaining() > 10.0:
        h_opt, obj_opt = _coordinate_descent(best_x, target_sum, n_iters=1000, lr_init=0.01)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = h_opt
    
    # Phase 3: More subgradient iterations from best
    if _time_remaining() > 15.0:
        h_opt, obj_opt = _subgradient_ascent(best_x, target_sum, n_iters=2000, lr_init=0.005)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = h_opt
    
    # Phase 4: Simulated annealing
    if _time_remaining() > 20.0:
        h_opt, obj_opt = _anneal_subgradient(best_x, target_sum, n_total_iters=3000)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = h_opt
    
    # Phase 5: Final coordinate descent
    if _time_remaining() > 8.0:
        h_opt, obj_opt = _coordinate_descent(best_x, target_sum, n_iters=2000, lr_init=0.005)
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_x = h_opt
    
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
