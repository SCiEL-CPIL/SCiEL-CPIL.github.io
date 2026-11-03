import numpy as np
import time
from scipy.signal import fftconvolve

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis for FFT updates
_freqs = np.arange(_FSIZE // 2 + 1)
_basis = np.exp(-2j * np.pi * _freqs[None, :] * np.arange(N)[:, None] / _FSIZE)

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h
        # dS/dh_i = 2 * sum_k v_k h_{k-i}
        # This is 2 * cross_correlation(v, h)
        
        # Use FFT for cross-correlation
        # We want C[i] = sum_{m=0}^{N-1} v[m+i] h[m]
        
        # Pad v and h to size FSIZE_CORR.
        # FSIZE_CORR must be >= (2N-1) + N - 1 = 3N - 2.
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        # C[i] should be sum_n v[n+i] conj(h[n])
        # Since h is real, conj(h[n]) = h[n].
        # So C[i] = sum_n v[n+i] h[n].
        # This is exactly what we want for i=0..N-1.
        
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            # Update FFT incrementally
            delta = -2 * old_val * _basis[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses FFT incrementally for speedup with precomputed basis.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 5, 7, 11, 15]
    
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        # Adaptive: use larger steps in later passes
        step_base = max(1, (N - 15) // (100 - min(pass_count * 10, 60)))
        
        for bs in block_sizes:
            step = max(1, step_base)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                # Flip block
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                # Update FFT using precomputed basis
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * _basis[idx]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                integral = np.sum(h)
                if abs(integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (integral * integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    improved = True
                else:
                    h[start:start+bs] = block  # revert
    return h, best_val

def _gradient_descent(h_init, deadline):
    """
    Perform gradient descent on the soft surrogate, then round to signs.
    """
    h = h_init.copy().astype(float)
    s0 = np.sum(h)
    if abs(s0) < 1e-6:
        return None
    
    h = h / s0  # Normalize to sum 1
    
    best_h = h.copy()
    best_val = compute_raw_value(np.sign(h))
    
    lr = 0.01
    beta = 20.0
    
    for step in range(200):
        if time.time() > deadline:
            break
        
        val, grad = _soft_max_abs_and_grad(h, beta=beta)
        
        # Gradient step
        h = h - lr * grad
        
        # Project to unit sum
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        h = h / s
        
        # Check sign-based value
        h_sign = np.sign(h)
        h_sign[h_sign == 0] = 1
        sign_val = compute_raw_value(h_sign)
        
        if sign_val < best_val:
            best_val = sign_val
            best_h = h_sign.copy()
        
        # Decay learning rate and increase beta
        lr *= 0.95
        if step % 20 == 0 and step > 0:
            beta *= 1.5
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    # Chirp sequences
    t = np.arange(N)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(15):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # Zadoff-Chu like (for N=400, use prime nearby or just construct)
    # Zadoff-Chu: x[n] = exp(-j * pi * u * n * (n+1) / N)
    # We take real part and sign
    for u in [1, 3, 5, 7]:
        if u < N:
            phase = -np.pi * u * t * (t + 1) / N
            x_zc = np.sign(np.cos(phase))
            x_zc[x_zc == 0] = 1
            candidates.append(x_zc)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Smooth surrogate optimization from best candidates using analytical gradient
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 8 diverse starts
    top_starts = scored_candidates[:8]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        # Gradient descent phase
        h_grad = _gradient_descent(h_init, deadline - 5)
        if h_grad is not None:
            val_grad = compute_raw_value(h_grad)
            if val_grad < best_val:
                best_val = val_grad
                best_x = h_grad.copy()
        
        # Local search on the gradient result (or original if grad failed)
        h_ls = h_grad if h_grad is not None else h_init
        h_flip, val_flip = _local_search_flip(h_ls, max_passes=10, deadline=deadline - 5)
        if val_flip < best_val:
            best_val = val_flip
            best_x = h_flip.copy()
        
        h_block, val_block = _local_search_block(h_flip, deadline=deadline - 5)
        if val_block < best_val:
            best_val = val_block
            best_x = h_block.copy()
    
    # 5. Random restarts from best solution
    if time.time() < deadline - 10:
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            s_perturbed = best_x.copy()
            # Flip 3-8 random elements
            n_flips = np.random.randint(3, 9)
            for _ in range(n_flips):
                idx = np.random.randint(0, N)
                s_perturbed[idx] = -s_perturbed[idx]
            
            s_new, val_new = _local_search_flip(s_perturbed, max_iters=800)
            if val_new < best_val:
                best_val = val_new
                best_x = s_new.copy()
    
    # Fallback
    if best_x is None:
        best_x = np.ones(N)
        best_x[0] = -1.0
    
    return best_x
