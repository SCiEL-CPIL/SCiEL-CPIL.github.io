"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _get_fft_size(n):
    """Get the next power of 2 size for FFT of linear convolution of length n."""
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs_and_grad(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp, with gradient w.r.t. ac.
    Returns (value, grad_ac) where grad_ac has same shape as ac.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp < 1e-300:
        return m, np.zeros_like(ac)
    log_sum = np.log(sum_exp)
    value = m + log_sum / beta
    # Gradient: d/dac_i [m + (1/beta) * log(sum_j exp(beta*(|ac_j| - m)))]
    # = (1/beta) * exp(beta*(|ac_i| - m)) / sum_j exp(beta*(|ac_j| - m)) * beta * sign(ac_i)
    # = exp_scaled[i] / sum_exp * sign(ac_i)
    grad_ac = (exp_scaled / sum_exp) * np.sign(ac)
    return value, grad_ac

def _objective_smooth_and_grad(h, beta=100.0):
    """
    Smooth objective with analytical gradient.
    f(h) = soft_max(|ac(h)|) / (sum h)^2
    where ac(h) = h * h (autoconvolution)
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    sm, grad_sm = _soft_max_abs_and_grad(ac, beta)
    
    # f = sm / s^2
    # df/dh_i = (grad_sm * dac/dh_i) / s^2 - 2*sm * (1/s^3)
    # dac/dh_i = 2 * (h convolved with delta_i) = 2 * h shifted
    # ac[k] = sum_j h[j] * h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (for linear convolution)
    
    # Compute gradient of ac w.r.t. h
    n = len(h)
    ac_len = 2*n - 1
    grad_ac_h = np.zeros((n, ac_len))
    # ac[k] = sum_{j} h[j] * h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] if both indices valid
    for k in range(ac_len):
        # d(ac[k])/d(h[i]) = h[k-i] + h[i-k]
        # For fixed k, as function of i:
        # term1: h[k-i] valid when 0 <= k-i < n => k-n < i <= k
        # term2: h[i-k] valid when 0 <= i-k < n => k <= i < k+n
        idx1 = np.arange(max(0, k-n+1), min(n, k+1))
        if len(idx1) > 0:
            j1 = k - idx1
            grad_ac_h[idx1, k] += h[j1]
        idx2 = np.arange(max(0, k), min(n, k+n))
        if len(idx2) > 0:
            j2 = idx2 - k
            grad_ac_h[idx2, k] += h[j2]
    
    # grad_f = (grad_sm @ grad_ac_h) / s^2 - 2*sm/s^3 * ones
    # grad_sm has shape (ac_len,), grad_ac_h has shape (n, ac_len)
    # grad_sm @ grad_ac_h has shape (n,)
    grad_f = (grad_sm @ grad_ac_h) / (s * s) - 2.0 * sm / (s * s * s) * np.ones(n)
    
    value = sm / (s * s)
    return value, grad_f

def _objective_smooth(h, beta=100.0):
    """Smooth objective for optimization (value only)."""
    val, _ = _objective_smooth_and_grad(h, beta)
    return val

def _gen_gaussian(n, sigma_ratio=0.25):
    """Generate a Gaussian sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a pseudo-random Barker-like sequence."""
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode='same')
    return smooth

def _local_search_flip(h, deadline=None, max_passes=5):
    """
    Local search with single-element sign flips and small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # Try flipping signs
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        # Try small perturbations
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            orig = h[i]
            for factor in [1 + step, 1 - step]:
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            else:
                h[i] = orig
        
        if not improved:
            break
            
    return h, best_val

def _local_search_block(h, block_size=5, deadline=None, max_passes=3):
    """
    Local search with block sign flips.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping blocks of consecutive elements
        for start in range(0, N - block_size + 1, block_size // 2):
            if deadline is not None and time.time() > deadline:
                break
            end = min(start + block_size, N)
            h[start:end] = -h[start:end]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[start:end] = -h[start:end]
        
        # Random block flips
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
            start = rng.integers(0, max(1, N - block_size))
            end = min(start + block_size, N)
            h[start:end] = -h[start:end]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[start:end] = -h[start:end]
        
        if not improved:
            break
            
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Diverse initializations
    inits = []
    inits.append(np.ones(N))
    inits.append(_gen_gaussian(N))
    inits.append(_gen_gaussian(N, sigma_ratio=0.15))
    inits.append(_gen_chirp(N))
    inits.append(_gen_barker_like(N))
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    inits.append(_gen_random_smoothed(N, seed=3, kernel_size=5))
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(2 * np.pi * x / N))
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    inits.append(np.sin(np.pi * x / N))
    
    for h_init in inits:
        if time.time() > deadline - 25:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            # Multi-stage beta optimization with analytical gradients
            betas = [30, 60, 120, 250]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 15:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def obj_grad(x):
                    val, grad = _objective_smooth_and_grad(x, beta=beta)
                    return val, grad
                
                res = minimize(obj_grad, current_h, method='L-BFGS-B', 
                               jac=True,
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6})
                
                h_opt = res.x
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                current_h = h_opt.copy()
                
        except Exception:
            pass

    # Local search refinement
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_flip(best_h, deadline=deadline-8, max_passes=6)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Block search
    if time.time() < deadline - 10:
        h_block, val_block = _local_search_block(best_h, block_size=7, deadline=deadline-5, max_passes=4)
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Final gradient polish with high beta
    if time.time() < deadline - 5:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=500.0)
                
                def obj_polish_grad(x):
                    val, grad = _objective_smooth_and_grad(x, beta=500.0)
                    return val, grad
                
                res = minimize(obj_polish_grad, h_polish_init, method='L-BFGS-B', 
                               jac=True,
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-7})
                
                h_polish = res.x
                if abs(np.sum(h_polish)) > 1e-6:
                    val_polish = _score(h_polish)
                    if val_polish < best_score:
                        best_score = val_polish
                        best_h = h_polish.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
