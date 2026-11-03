"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_incremental_update(A, h, k):
    """
    Update autoconvolution A after flipping h[k].
    """
    h_k = h[k]
    A_new = A.copy()
    A_new[k:k+N] -= 4 * h_k * h
    if 2*k < 2*N - 1:
        A_new[2*k] += 4.0
    return A_new

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_and_grad(h, beta):
    """
    Compute soft-max of |autoconv| and its gradient w.r.t. h.
    Uses log-sum-exp for numerical stability.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        w = np.zeros_like(ac)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        log_sum = np.log(np.sum(exp_scaled))
        S = m + log_sum / beta
        w = exp_scaled / np.sum(exp_scaled)
    
    # Gradient: dS/dh_k = sum_j w_j * sign(ac_j) * (h_{j-k} + h_{j+k})
    # = 2 * circular_conv(w * sign(ac), h) truncated to n
    v = w * np.sign(ac)
    V = np.fft.rfft(v, fsize)
    C = np.fft.irfft(V * H, fsize)
    dS_dh = 2 * C[:n]
    
    return S, dS_dh

def _smooth_obj_with_grad(x, beta):
    """
    Smooth surrogate objective with gradient.
    f(x) = S(x) / (sum(x))^2 where S is soft-max of |autoconv|
    """
    S, dS_dh = _soft_max_abs_and_grad(x, beta)
    s = np.sum(x)
    
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(x)
    
    f = S / (s * s)
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    
    return f, grad

def _fast_local_search(h, max_passes=1000, deadline=None):
    """
    Fast local search with single-bit flips, block moves, and pair swaps.
    """
    h = h.copy()
    s = np.sum(h)
    A = _autoconv_fft(h)
    current_score = np.max(np.abs(A)) / (s * s)
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single-bit flips
        for k in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h_k = h[k]
            A_new = _autoconv_incremental_update(A, h, k)
            s_new = s - 2 * h_k
            
            if abs(s_new) < 1e-9:
                continue
                
            score_new = np.max(np.abs(A_new)) / (s_new * s_new)
            
            if score_new < current_score * (1 - 1e-15):
                h[k] = -h_k
                A = A_new
                s = s_new
                current_score = score_new
                improved = True
        
        # Block moves: flip 2-4 consecutive bits
        if p % 3 == 0:
            for block_size in [2, 3, 4]:
                for start in range(0, N - block_size + 1, 1):
                    if deadline is not None and time.time() > deadline:
                        break
                    
                    h_trial = h.copy()
                    s_trial = s
                    for k in range(start, start + block_size):
                        h_trial[k] = -h_trial[k]
                        s_trial -= 2 * h[k]
                    
                    if abs(s_trial) < 1e-9:
                        continue
                    
                    A_trial = _autoconv_fft(h_trial)
                    score_trial = np.max(np.abs(A_trial)) / (s_trial * s_trial)
                    
                    if score_trial < current_score * (1 - 1e-15):
                        h = h_trial
                        s = s_trial
                        A = A_trial
                        current_score = score_trial
                        improved = True
        
        # Pair swap: flip two non-adjacent positions
        if p % 5 == 0:
            for i in range(0, N, 2):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(i + 2, min(i + 20, N), 3):
                    if deadline is not None and time.time() > deadline:
                        break
                    
                    h_trial = h.copy()
                    s_trial = s
                    h_trial[i] = -h_trial[i]
                    h_trial[j] = -h_trial[j]
                    s_trial -= 2 * (h[i] + h[j])
                    
                    if abs(s_trial) < 1e-9:
                        continue
                    
                    A_trial = _autoconv_fft(h_trial)
                    score_trial = np.max(np.abs(A_trial)) / (s_trial * s_trial)
                    
                    if score_trial < current_score * (1 - 1e-15):
                        h = h_trial
                        s = s_trial
                        A = A_trial
                        current_score = score_trial
                        improved = True
        
        if not improved:
            break
    
    return h, current_score

def _high_precision_polish(h, deadline):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    h = h.copy()
    s = np.sum(h)
    A = _autoconv_fft(h)
    current_score = np.max(np.abs(A)) / (s * s)
    
    for pass_num in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for k in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h_k = h[k]
            A_new = _autoconv_incremental_update(A, h, k)
            s_new = s - 2 * h_k
            
            if abs(s_new) < 1e-9:
                continue
                
            score_new = np.max(np.abs(A_new)) / (s_new * s_new)
            
            threshold = current_score * (1 - 1e-14 * (pass_num + 1))
            
            if score_new < threshold:
                h[k] = -h_k
                A = A_new
                s = s_new
                current_score = score_new
                improved = True
        
        if not improved:
            break
    
    return h, current_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    x = np.arange(N)
    inits.append(np.ones(N))
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    phase = np.pi * x**2 / N
    inits.append(np.cos(phase))
    inits.append(np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)]))
    inits.append(np.sin(np.pi * x / N))
    inits.append(np.cos(2 * np.pi * x / N))
    
    rng = np.random.default_rng(42)
    for i in range(8):
        noise = rng.standard_normal(N)
        kernel_size = 5 + i
        kernel = np.ones(kernel_size)/kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    for i in range(8):
        inits.append(rng.choice([1.0, -1.0], N))
    
    # Structured random: alternating blocks
    for block_len in [4, 8, 16]:
        pattern = np.array([1.0 if (i // block_len) % 2 == 0 else -1.0 for i in range(N)])
        inits.append(pattern)

    # Optimization loop using smooth surrogate with beta annealing
    for h_init in inits:
        if time.time() > deadline - 35:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            for beta in [5, 15, 40, 80, 150]:
                if time.time() > deadline - 15:
                    break
                
                res = minimize(lambda x: _smooth_obj_with_grad(x, beta), h_norm, 
                             method='L-BFGS-B', jac=True,
                             options={'maxiter': 200, 'disp': False, 'gtol': 1e-7})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                    
                # Also try binarized version
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = _score(h_bin)
                if val_bin < best_score:
                    best_score = val_bin
                    best_h = h_bin.copy()
                    
                # Try thresholded version
                h_thresh = np.where(h_opt > 0, 1.0, -1.0)
                val_thresh = _score(h_thresh)
                if val_thresh < best_score:
                    best_score = val_thresh
                    best_h = h_thresh.copy()
                    
        except Exception:
            pass

    # Local search refinement
    if time.time() < deadline - 8:
        h_work = best_h.copy()
        h_refined, val_refined = _fast_local_search(h_work, max_passes=1500, deadline=deadline-3)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # High-precision polish
    if time.time() < deadline - 1:
        h_polished, val_polished = _high_precision_polish(best_h.copy(), deadline-0.5)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
