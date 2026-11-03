"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
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

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2"""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _local_search_flip_fast(h, max_passes=10, deadline=None):
    """
    Fast local optimization: try flipping each element's sign using incremental FFT.
    Works on binary {-1, 1} sequences.
    """
    h = h.copy()
    # Ensure binary
    h = np.sign(h)
    h[h == 0] = 1.0
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    best_val = _score(h)
    
    # Precompute FFT size and initial spectrum
    fsize, size = _get_fft_size(N)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flip h[i]: h[i] -> -h[i], so delta = -2*h[i]
            old_h_i = h[i]
            delta = -2.0 * old_h_i
            h[i] = -old_h_i
            
            # Update FFT: H_new[k] = H_old[k] + delta * exp(-2j*pi*i*k/fsize)
            k_vals = np.arange(fsize // 2 + 1)
            phase = np.exp(-2j * np.pi * i * k_vals / fsize)
            H += delta * phase
            
            # Recompute autoconvolution from updated spectrum
            ac = np.fft.irfft(H * H, fsize)[:size]
            new_max_abs = np.max(np.abs(ac))
            
            # Update sum
            new_s = s - 2.0 * old_h_i
            if abs(new_s) < 1e-9:
                new_val = 1e10
            else:
                new_val = new_max_abs / (new_s * new_s)
            
            if new_val < best_val:
                best_val = new_val
                improved = True
                s = new_s
            else:
                # Revert
                h[i] = old_h_i
                H -= delta * phase
    
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3, 5, 7, 10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    Helps escape local minima of single-bit flips.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1.0
    best_val = _score(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 80)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
    return h, best_val

def _binarize(h):
    """Convert to binary {-1, 1} sequence."""
    h = np.sign(h)
    h[h == 0] = 1.0
    return h

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)

    # Baseline: all ones
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate diverse binary candidates
    candidates = []
    t = np.arange(N)
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Chirp sequences (quadratic phase)
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1.0
        candidates.append(x_chirp)
    
    # 3. Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1.0
        candidates.append(x_lin)
    
    # 4. Barker-like structured
    x_barker = np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)])
    candidates.append(x_barker)
    
    # 5. Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1.0
    candidates.append(x_gold)
    
    # 6. Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # 7. Smoothed random (continuous, then binarize)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        noise = rng.standard_normal(N)
        kernel = np.ones(10) / 10
        smoothed = np.convolve(noise, kernel, mode='same')
        x_smooth = _binarize(smoothed)
        candidates.append(x_smooth)
    
    # 8. Gaussian-windowed random
    for seed in [11, 22, 33]:
        noise = rng.standard_normal(N)
        mu = N / 2
        sigma = N * 0.25
        window = np.exp(-(t - mu)**2 / (2 * sigma**2))
        x_gauss = _binarize(noise * window)
        candidates.append(x_gauss)
    
    # Evaluate all candidates
    for x in candidates:
        val = _score(x)
        if val < best_score:
            best_score = val
            best_h = x.copy()

    # Smooth surrogate optimization from top candidates
    # Find top 5 diverse starts
    scored_candidates = []
    for x in candidates:
        val = _score(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:5]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 30:
            break
        
        # Normalize to sum 1 for optimization
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Multi-stage beta optimization
        for beta in [15, 30, 60, 100]:
            if time.time() > deadline - 15:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                    
                    # Also try sign-binarized version
                    h_bin = _binarize(h_opt)
                    val_bin = _score(h_bin)
                    if val_bin < best_score:
                        best_score = val_bin
                        best_h = h_bin.copy()
            except Exception:
                pass

    # Local search refinement on best result
    if time.time() < deadline - 15:
        # Binarize if not already binary
        h_work = _binarize(best_h)
        
        # Single-bit flip local search
        h_work, val_work = _local_search_flip_fast(h_work, max_passes=10, deadline=deadline - 8)
        if val_work < best_score:
            best_score = val_work
            best_h = h_work.copy()
        
        # Block flip local search
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 3, 
                                                    block_sizes=[3, 5, 7, 10])
            if val_work < best_score:
                best_score = val_work
                best_h = h_work.copy()
    
    # Additional random restarts with local search if time remains
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip_fast(x_rand, max_passes=3, deadline=deadline - 5)
            if val_work < best_score:
                best_score = val_work
                best_h = h_work.copy()
    
    # Final polish: one more flip pass on the best result
    if time.time() < deadline - 2:
        h_final, val_final = _local_search_flip_fast(_binarize(best_h), max_passes=5, deadline=deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
