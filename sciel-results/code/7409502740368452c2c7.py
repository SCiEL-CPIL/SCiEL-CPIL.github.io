"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

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

def _gen_sinc_window(n):
    """Generate a sinc-windowed sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * 0.15
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss

def _gen_alternating_gauss(n):
    """Alternating signs with Gaussian envelope."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * 0.3
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    signs = np.where(x % 2 == 0, 1.0, -1.0)
    return gauss * signs

def _local_search(h, max_passes=5, deadline=None):
    """
    Local search: try flipping signs or small perturbations.
    Uses the exact score function.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    step_sizes = [0.2, 0.1, 0.05, 0.02]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # 1. Try flipping signs
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
        
        # 2. Try small multiplicative perturbations
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

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random Smoothed
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    
    # 6. Cosine window
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    # 7. Alternating signs with Gaussian envelope
    inits.append(_gen_alternating_gauss(N))
    
    # 8. Sinc window
    inits.append(_gen_sinc_window(N))
    
    # 9. Wider Gaussian
    inits.append(_gen_gaussian(N, sigma_ratio=0.15))
    
    # 10. Narrow Gaussian
    inits.append(_gen_gaussian(N, sigma_ratio=0.35))
    
    for h_init in inits:
        if time.time() > deadline - 30:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            betas = [50, 100, 200]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, current_h, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
                
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

    # Final Local Search on the best candidate found so far
    if time.time() < deadline - 10:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
    # Optional: One more gradient polish on the refined result with high beta
    if time.time() < deadline - 5:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=500.0)
                
                res = minimize(obj_polish, h_polish_init, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
                
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
