"""Autoconvolution peak suppression via FFT-accelerated log-sum-exp optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
DEADLINE = 143.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _autoconv(h):
    """Compute autoconvolution efficiently using FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]

def _smooth_objective(h, beta=100.0):
    """Smooth surrogate using log-sum-exp approximation of max(conv)."""
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    shifted = conv - max_c
    # Numerical stability
    max_shifted = float(np.max(shifted))
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    log_sum = max_c_adj + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(h, beta=100.0):
    """
    Gradient of the smooth surrogate computed via FFT.
    
    lse = max_c + (1/beta) * log(sum(exp(beta * (conv - max_c))))
    d_lse/dh_m = sum_k w_k * d_conv_k/dh_m where w_k = softmax(beta*(conv-max_c))
    d_conv_k/dh_m = 2*h_{k-m} if 0<=k-m<N else 0
    So d_lse/dh_m = 2 * sum_j w_{m+j} * h_j  (cross-correlation)
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    shifted = conv - max_c
    max_shifted = float(np.max(shifted))
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    
    # Softmax weights
    exp_vals = np.exp(beta * shifted)
    weight_sum = float(np.sum(exp_vals))
    weights = exp_vals / weight_sum
    
    lse = max_c_adj + np.log(weight_sum) / beta
    
    # Cross-correlation: d_lse_dh[m] = 2 * sum_j weights[m+j] * h[j]
    # Use FFT: pad to length >= 2N-1 + N - 1 = 3N-2, use 3N for safety
    n_corr = 1
    while n_corr < (2 * N - 1) + N - 1:
        n_corr *= 2
    
    W = np.fft.rfft(weights, n=n_corr)
    H = np.fft.rfft(h, n=n_corr)
    
    # cross[m] = sum_j weights[m+j] * h[j]
    # This is the cross-correlation of weights and h.
    # In FFT terms: ifft(conj(FFT(h)) * FFT(weights)) gives correlation
    # But we need: sum_j w[m+j] h[j] = (w * h_flipped)[m] where h_flipped is reversed
    # Actually: sum_j w[m+j] h[j] = sum_j w[m+j] h[j]
    # Let's define: for the convolution approach,
    # (w conv h_flipped)[m] = sum_j w[m-j] h_flipped[j] = sum_j w[m-j] h[N-1-j]
    # That's not quite right either.
    
    # Direct approach via FFT:
    # Define A = weights (length 2N-1), B = h (length N)
    # We want C[m] = sum_j A[m+j] * B[j] for j=0..N-1 where m+j < 2N-1
    # C[m] = sum_{j=0}^{N-1} A[m+j] B[j]
    # Let A_rev[k] = A[2N-2-k], then A[m+j] = A_rev[2N-2-m-j]
    # C[m] = sum_j A_rev[2N-2-m-j] B[j] = (A_rev * B)[2N-2-m]  (convolution)
    # So C = reverse of (A_rev * B) appropriately indexed
    
    A_rev = weights[::-1]  # length 2N-1
    # Convolution of A_rev and B: length (2N-1) + N - 1 = 3N-2
    conv_result = np.fft.irfft(W * np.fft.rfft(h, n=n_corr), n=n_corr)[:3*N-2]
    
    # conv_result[k] = sum_j A_rev[k-j] * B[j] = sum_j A[2N-2-(k-j)] * B[j]
    # = sum_j A[2N-2-k+j] * B[j]
    # We want C[m] = sum_j A[m+j] B[j], so set 2N-2-k = m, i.e., k = 2N-2-m
    # C[m] = conv_result[2N-2-m]
    
    d_lse_dh = np.zeros(N)
    for m in range(N):
        k = 2 * N - 2 - m
        if 0 <= k < len(conv_result):
            d_lse_dh[m] = 2.0 * conv_result[k]
        else:
            d_lse_dh[m] = 0.0
    
    # f = 2N * lse / s^2, df/dh_m = 2N * (d_lse_dh[m]*s - 2*lse) / s^3
    f_inner = 2.0 * N * lse / (s * s)
    df_dh = 2.0 * N * (d_lse_dh * s - 2.0 * lse) / (s * s * s)
    
    if f_inner >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _generate_initializations():
    """Generate diverse initializations targeting flat autoconvolution."""
    inits = []
    i_arr = np.arange(N)
    t = i_arr / N
    
    # Chirp signals - these tend to flatten autoconvolution
    for alpha in [1, 2, 3, 5, 8, 13]:
        h = np.cos(2 * np.pi * (t + alpha * t**2))
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
        inits.append((-h).copy())
    
    # Wigner-like: exp(-a(t-0.5)^2) * cos(2*pi*b*t)
    for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for b in [1, 2, 3, 5, 8]:
            h = np.exp(-a * (t - 0.5)**2) * np.cos(2 * np.pi * b * t)
            s = np.sum(h)
            if abs(s) < 1e-10:
                h += 0.05
            inits.append(h.copy())
    
    # Multi-frequency with phase shifts
    freq_combos = [
        [1, 3, 5, 7],
        [2, 5, 8, 11],
        [1, 4, 7, 10, 13],
        [3, 6, 9, 12],
        [1, 2, 3, 4, 5, 6, 7],
        [2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9, 11],
        [5, 10, 15, 20],
        [1, 5, 10, 15, 25],
        [2, 7, 12, 17],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 7, 11, 15, 19],
        [4, 9, 14, 19, 24],
        [1, 3, 7, 11, 15],
    ]
    
    for freqs in freq_combos:
        h = np.zeros(N)
        for f in freqs:
            h += np.cos(2 * np.pi * f * t)
            h += 0.3 * np.sin(2 * np.pi * f * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Alternating sign versions
    for freqs in freq_combos[:8]:
        h = np.zeros(N)
        for i, f in enumerate(freqs):
            sign = 1 if i % 2 == 0 else -1
            h += sign * np.cos(2 * np.pi * f * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Random smooth perturbations
    rng = np.random.RandomState(42)
    from scipy.ndimage import gaussian_filter1d
    for _ in range(15):
        h_r = rng.uniform(-1, 1, N)
        h_r = gaussian_filter1d(h_r, sigma=rng.uniform(5, 40))
        s = np.sum(h_r)
        if abs(s) < 1e-10:
            h_r += 0.05
        inits.append(h_r.copy())
    
    # Sine with various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 11, 13, 17, 21]:
        h = np.sin(2 * np.pi * freq * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Cosine with various frequencies  
    for freq in [1, 2, 3, 4, 5, 7, 11, 13, 17, 21]:
        h = np.cos(2 * np.pi * freq * t)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    # Piecewise constant with random breakpoints
    rng2 = np.random.RandomState(123)
    for _ in range(8):
        n_breaks = rng2.randint(5, 20)
        breaks = sorted(rng2.choice(np.arange(1, N), size=n_breaks, replace=False))
        h = np.zeros(N)
        for i in range(len(breaks) + 1):
            start = breaks[i-1] if i > 0 else 0
            end = breaks[i] if i < len(breaks) else N
            h[start:end] = rng2.uniform(-1, 1)
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h.copy())
    
    return inits

def _optimize_from(h_init, t0, deadline):
    """Run multi-beta optimization from a single starting point."""
    best_h = h_init.copy()
    best_val = _eval_value(h_init)
    
    s = np.sum(h_init)
    if abs(s) < 1e-10:
        h_init = h_init + 0.05
    
    # Multi-scale beta annealing
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    current_h = h_init.copy()
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        try:
            res = minimize(
                lambda x, b=beta: _smooth_objective(x, b),
                current_h,
                jac=lambda x, b=beta: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-10}
            )
            
            if time.time() > deadline - 2:
                break
            
            h_candidate = res.x
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                v = _eval_value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
                    current_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    # --- Phase 1: Generate and evaluate diverse initializations ---
    candidates = _generate_initializations()
    
    cand_vals = []
    for h in candidates:
        v = _eval_value(h)
        if np.isfinite(v):
            cand_vals.append((v, h))
    
    cand_vals.sort(key=lambda x: x[0])
    
    # Keep top 15 candidates
    top_candidates = [h for _, h in cand_vals[:15]]
    
    if top_candidates:
        v_top = _eval_value(top_candidates[0])
        if v_top < best_val:
            best_val = v_top
            best_h = top_candidates[0].copy()
    
    # --- Phase 2: Optimize from top initializations ---
    for h_init in top_candidates[:10]:
        if elapsed() > DEADLINE - 20:
            break
        
        h_opt, v_opt = _optimize_from(h_init, t0, DEADLINE - 5)
        if v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
    
    # --- Phase 3: High-precision polish from best ---
    if elapsed() < DEADLINE - 15:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 0.01
        
        for beta in [500.0, 1000.0, 2000.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    h0,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 800, 'gtol': 1e-12}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = res.x
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    v = _eval_value(h_candidate)
                    if v < best_val:
                        best_val = v
                        best_h = h_candidate.copy()
                        h0 = h_candidate.copy()
            except Exception:
                pass
    
    # --- Phase 4: Try remaining candidates if time permits ---
    if elapsed() < DEADLINE - 10:
        for h_init in top_candidates[10:]:
            if elapsed() > DEADLINE - 8:
                break
            
            h_opt, v_opt = _optimize_from(h_init, t0, DEADLINE - 5)
            if v_opt < best_val:
                best_val = v_opt
                best_h = h_opt.copy()
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
