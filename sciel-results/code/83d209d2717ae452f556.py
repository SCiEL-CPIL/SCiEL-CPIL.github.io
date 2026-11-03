import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = _fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
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
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a Barker-like sequence."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_golden_ratio(n):
    """Generate a sequence based on golden ratio properties."""
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    frac = (i * phi) % 1.0
    return 2 * frac - 1

def _gradient_polish(h, deadline, betas=[100, 200, 400]):
    """
    Gradient-based final polish using increasing beta for smoother -> sharper max approximation.
    This is much faster than coordinate descent local search.
    """
    best_h = h.copy()
    best_val = _score(best_h)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        try:
            res = minimize(obj, best_h, method='L-BFGS-B', 
                           options={'maxiter': 200, 'disp': False, 'gtol': 1e-7, 'ftol': 1e-10})
            h_opt = res.x
            val = _score(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _coordinate_descent_fast(h, deadline, max_rounds=3):
    """
    Fast coordinate descent: only try a subset of coordinates per round to stay within time budget.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Shuffle coordinates for diversity
        rng = np.random.default_rng(round_num + 1000)
        indices = np.random.permutation(n)
        
        improved_this_round = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            old_val = h[idx]
            # Try a few perturbation sizes
            for delta in [0.01, -0.01, 0.05, -0.05]:
                h[idx] = old_val + delta
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved_this_round = True
                    break
            else:
                h[idx] = old_val  # revert
        
        if not improved_this_round:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(8):
        noise = rng.standard_normal(N)
        kernel_size = 5 + i * 2
        kernel = np.ones(kernel_size)/kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
    
    # 8. Golden ratio sequence
    inits.append(_gen_golden_ratio(N))
    
    # 9. More random sequences with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append(rng.standard_normal(N))
    
    # 10. Alternating patterns with varying periods
    for period in [3, 5, 7, 11]:
        h_pat = np.zeros(N)
        for i in range(N):
            h_pat[i] = 1.0 if (i % period) < period // 2 else -1.0
        inits.append(h_pat)
    
    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Start with moderate beta, increase for refinement
            for beta in [15, 30, 60]:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            continue
    
    # Gradient-based final polish on the best solution found
    if time.time() < deadline - 10:
        best_h, best_score = _gradient_polish(best_h, deadline - 5)
    
    # Fast coordinate descent if time remains
    if time.time() < deadline - 2:
        best_h, best_score = _coordinate_descent_fast(best_h, deadline, max_rounds=2)
    
    return best_h
