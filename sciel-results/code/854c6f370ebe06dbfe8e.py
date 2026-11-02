import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
DEADLINE = 145  # Safety margin from 150s limit
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    """Compute the ratio: (sum h^2) / (max autoconvolution) for normalized h."""
    h = np.array(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    total = np.sum(h)
    if total == 0:
        return 0.0
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / max_conv

def _reweighting_optimize(h_init, support, max_iter=200, tol=1e-10):
    """
    Iterative reweighting on fixed support to find optimal amplitudes.
    For fixed support, we maximize sum(h_i^2) / max_k(sum_{i+j=k} h_i*h_j)
    subject to sum(h_i) = 1, h_i >= 0.
    
    Uses multiplicative updates: h_i <- h_i * (2*h_i / max_conv) / sum(h_j * 2*h_j / max_conv)
    But we need to handle the max carefully. Instead, we use coordinate descent
    on the log-likelihood style objective.
    """
    h = np.zeros(N)
    h[support] = h_init[support]
    total = np.sum(h)
    if total == 0:
        return h
    
    for _ in range(max_iter):
        if _time_remaining() < 5:
            break
            
        h_norm = h / np.sum(h)
        conv = _autoconvolution_fft(h_norm)
        max_conv = np.max(conv)
        
        if max_conv <= 0:
            break
            
        # Gradient of score w.r.t. h_i (before normalization)
        # score = sum(h^2)/sum(h) / max_conv = (sum h_i^2) / (sum h * max_conv)
        # Let's use a different approach: optimize directly on normalized h
        # with constraint sum(h) = 1
        
        # Coordinate descent: for each i in support, find optimal h_i
        improved = False
        for idx in support:
            old_val = h[idx]
            if old_val < 1e-15:
                continue
                
            # Try a few steps up and down
            best_local = old_val
            best_local_score = _score(h)
            
            for step_factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
                trial_val = old_val * step_factor
                if trial_val < 0:
                    continue
                h_trial = h.copy()
                h_trial[idx] = trial_val
                s_trial = _score(h_trial)
                if s_trial > best_local_score + 1e-12:
                    best_local_score = s_trial
                    best_local = trial_val
            
            if abs(best_local - old_val) > 1e-12:
                h[idx] = best_local
                improved = True
        
        if not improved:
            break
    
    return h

def _coordinate_polish(h, support, max_iter=50, tol=1e-12):
    """High-precision coordinate polish using golden section search."""
    h = h.copy()
    for _ in range(max_iter):
        if _time_remaining() < 3:
            break
            
        improved = False
        for idx in support:
            old_val = h[idx]
            if old_val < 1e-15:
                continue
            
            # Golden section search on [0, old_val * 3]
            a = 0.0
            b = max(old_val * 3, 1e-10)
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = _score(np.where(np.arange(N) == idx, c, h))
            fd = _score(np.where(np.arange(N) == idx, d, h))
            
            for _ in range(30):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _score(np.where(np.arange(N) == idx, c, h))
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _score(np.where(np.arange(N) == idx, d, h))
                
                if b - a < tol:
                    break
            
            opt_val = (a + b) / 2
            if abs(opt_val - old_val) > 1e-10:
                h[idx] = opt_val
                improved = True
        
        if not improved:
            break
    
    return h

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guesses
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Candidate 1: [2/3, 1/3, 0...]
    h1 = np.zeros(N)
    h1[0] = 2/3
    h1[1] = 1/3
    s1 = _score(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1.copy()
        
    # Candidate 2: [2/3, 1/3, 1/4, 0...]
    h2 = np.zeros(N)
    h2[0] = 2/3
    h2[1] = 1/3
    h2[2] = 1/4
    h2 = h2 / np.sum(h2)
    s2 = _score(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2.copy()
    
    # Candidate 3: Uniform on 2 points
    h3 = np.zeros(N)
    h3[0] = 0.5
    h3[1] = 0.5
    s3 = _score(h3)
    if s3 > best_score:
        best_score = s3
        best_h = h3.copy()
    
    # Candidate 4: [1, 0, 0, ..., 0] (degenerate)
    h4 = np.zeros(N)
    h4[0] = 1.0
    s4 = _score(h4)
    if s4 > best_score:
        best_score = s4
        best_h = h4.copy()
    
    # Generate diverse support sets and optimize
    np.random.seed(42)
    
    # Predefined good supports
    support_candidates = []
    
    # Small supports near origin
    for size in range(2, 8):
        for start in range(0, N - size + 1, 2):
            support_candidates.append(list(range(start, start + size)))
    
    # Random supports
    for _ in range(20):
        size = np.random.randint(2, 10)
        support = np.random.choice(N, size, replace=False)
        support.sort()
        support_candidates.append(list(support))
    
    # Optimize on each support
    for support in support_candidates:
        if _time_remaining() < 10:
            break
        
        # Initialize with random weights on support
        for init_type in range(3):
            if _time_remaining() < 5:
                break
            
            h_init = np.zeros(N)
            if init_type == 0:
                # Uniform
                h_init[support] = 1.0 / len(support)
            elif init_type == 1:
                # Random
                h_init[support] = np.random.rand(len(support))
                h_init /= np.sum(h_init)
            else:
                # Decaying
                weights = np.exp(-np.arange(len(support)) * 0.3)
                h_init[support] = weights
                h_init /= np.sum(h_init)
            
            # Run reweighting
            h_opt = _reweighting_optimize(h_init, support, max_iter=100)
            s_opt = _score(h_opt)
            
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
                # Polish with coordinate descent
                h_polished = _coordinate_polish(h_opt, support, max_iter=30)
                s_polished = _score(h_polished)
                if s_polished > best_score:
                    best_score = s_polished
                    best_h = h_polished.copy()
    
    # Final polish on best solution
    if _time_remaining() > 10:
        support_best = list(np.where(best_h > 1e-10)[0])
        if len(support_best) > 0:
            h_final = _coordinate_polish(best_h, support_best, max_iter=50)
            s_final = _score(h_final)
            if s_final > best_score:
                best_score = s_final
                best_h = h_final.copy()
    
    # Ensure non-negative and normalized
    best_h = np.abs(best_h)
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
