"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    # Autoconvolution is convolution of h with h
    # In frequency domain: H * H
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    # Use a quadratic phase
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n):
    """Generate a windowed sinc sequence."""
    i = np.arange(n)
    center = n / 2
    # Sinc function centered
    x = (i - center)
    # Avoid division by zero
    sinc = np.ones_like(x)
    nonzero = x != 0
    sinc[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    # Apply a Hamming window to reduce sidelobes in frequency (or time?)
    # We want h to be spread out to flatten h*h.
    # A Gaussian is a good spread-out function.
    sigma = n / 4
    return np.exp(-0.5 * (x / sigma) ** 2)

def _local_search(h, max_passes=20, deadline=None):
    """
    Local search: try flipping signs or small perturbations with adaptive step size.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    step_sizes = [0.1, 0.05, 0.02, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step_idx = min(pass_num // 5, len(step_sizes) - 1)
        step = step_sizes[step_idx]
        
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
        
        # Try multiplicative perturbations
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
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = np.ones(N) # Default: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.arange(N)
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Sinc (windowed)
    inits.append(_gen_sinc(N))
    
    # 5. Random smoothed
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(N)
    kernel = np.ones(10)/10
    smooth = np.convolve(noise, kernel, mode='same')
    inits.append(smooth)
    
    # 6. Alternating signs with Gaussian envelope
    x = np.arange(N)
    mu = N/2
    sigma = N/3
    gauss_env = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    # 7. Barker-like sequence (pseudo-random with low autocorrelation)
    barker_like = np.ones(N)
    for i in range(1, N):
        barker_like[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    inits.append(barker_like)
    
    # 8. Cosine-modulated chirp
    x = np.arange(N)
    chirp_cos = np.cos(np.pi * x**2 / N) * np.cos(2 * np.pi * x / N)
    inits.append(chirp_cos)
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Optimize with multiple beta values for smoothness
        try:
            for beta in [50, 100, 200]:
                if time.time() > deadline - 10:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search with more passes
    if time.time() < deadline - 5:
        h_refined, val_refined = _local_search(best_h, max_passes=15, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
