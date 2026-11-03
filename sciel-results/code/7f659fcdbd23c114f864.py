"""Rewritten program: gradient-based smooth surrogate optimization + block sign-flip local search."""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _fft_size(n):
    """Next power of 2 >= n."""
    s = 1
    while s < n:
        s *= 2
    return s

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fs = _fft_size(size)
    H = np.fft.rfft(h, fs)
    ac = np.fft.irfft(H * H, fs)
    return ac[:size]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=30.0):
    """Smooth approximation of max|ac| via log-sum-exp."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=30.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=30.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    
    ac = autoconv(h), so ac[k] = sum_i h[i]*h[k-i].
    d(ac[k])/d(h[j]) = h[k-j] + h[j-k] (for valid indices).
    
    For the autoconvolution via FFT: ac = ifft(fft(h)*fft(h)).
    Gradient of ac w.r.t. h: ifft(2 * fft(h) * fft(conj(h[::-1])))... 
    Actually simpler: ac = h * h (convolution). d(ac)/d(h[j]) is a sequence.
    
    Let me think more carefully. ac[m] = sum_j h[j] * h[m-j].
    d(ac[m])/d(h[j]) = h[m-j] + h[j-m] (if both indices valid, else just one term).
    
    This is equivalent to: d(ac)/d(h[j]) = h[m-j] evaluated at m, summed over m with weights.
    
    Actually, the gradient of soft_max_abs(ac) w.r.t. ac is:
    g_ac[m] = weights[m] * sign(ac[m])
    
    Then gradient w.r.t. h[j]:
    grad_h[j] = sum_m g_ac[m] * d(ac[m])/d(h[j])
    = sum_m g_ac[m] * (h[m-j] + h[j-m])  [with boundary handling]
    
    This can be computed via FFT:
    d(ac)/d(h[j]) for all j is essentially the convolution of g_ac with h.
    
    Specifically: sum_m g_ac[m] * h[m-j] = (g_ac * h)[j] (cross-correlation-like)
    And sum_m g_ac[m] * h[j-m] = (g_ac * h)[j] as well (same thing by symmetry of autoconv).
    
    Wait, let me be more careful. ac[m] = sum_j h[j] h[m-j].
    d(ac[m])/d(h[j]) = h[m-j] + h[j-m] (when both 0<=m-j<N and 0<=j-m<N, i.e., when |m-j|<N).
    
    So grad_h[j] = sum_m g_ac[m] * (h[m-j] + h[j-m])
    = sum_m g_ac[m] * h[m-j] + sum_m g_ac[m] * h[j-m]
    
    First term: let k = m-j, so m = k+j. Sum over k where 0<=k+j<N: g_ac[k+j] * h[k].
    This is a cross-correlation: (g_ac ⊛ h)[j] where (a ⊛ b)[j] = sum_k a[k+j]*b[k].
    
    Second term: let k = j-m, so m = j-k. Sum over k where 0<=j-k<N: g_ac[j-k] * h[k].
    This is also a cross-correlation: sum_k g_ac[j-k]*h[k] = (g_ac ⊛ h)[j] (same!).
    
    Wait no. First term: sum_k g_ac[k+j]*h[k]. Second term: sum_k g_ac[j-k]*h[k].
    These are different. First is cross-correlation of g_ac and h at lag j.
    Second is also cross-correlation but with g_ac[j-k] which is g_ac reflected.
    
    Actually, since ac is an autoconvolution, it's symmetric: ac[m] = ac[-m] (in the full sense).
    And g_ac should inherit some symmetry. Let me just compute it directly using FFT.
    
    The key insight: ac = ifft(|fft(h)|^2) [in the circular sense, but we use linear].
    
    For linear convolution: ac = ifft(fft(h, fs) * fft(h, fs)) truncated.
    
    d(ac)/d(h) in the frequency domain: 
    Let H = fft(h, fs). ac = ifft(H^2).
    d(ac)/d(h[j]) = ifft(2 * H * fft(e_j, fs)) where e_j is the unit vector at position j.
    fft(e_j) = exp(-2*pi*i*j*n/fs) for the DFT.
    
    So grad_h[j] = sum_m g_ac[m] * d(ac[m])/d(h[j])
    
    This is getting complex. Let me use a simpler approach: compute the gradient numerically
    or use the fact that for autoconvolution, the gradient can be computed as:
    
    grad_h = 2 * ifft(fft(g_ac, fs) * conj(fft(h, fs)))  [truncated to N]
    
    Let me verify: if ac = h*h (linear conv), then in frequency: Ac = H^2 (circular with padding).
    d(Ac)/d(H[j]) = 2*H[j] * d(H[j])/d(h) ... this doesn't quite work for linear conv.
    
    Let me just do it the straightforward way with FFT:
    
    For linear autoconvolution with zero-padding to size fs:
    ac = ifft(H * H)[:size] where H = rfft(h, fs)
    
    The gradient of ac w.r.t. h: 
    d(ac[m])/d(h[j]) involves the convolution kernel.
    
    Simplest correct approach: 
    grad_h[j] = sum_m g_ac[m] * (h[(m-j) % N] if valid + h[(j-m) % N] if valid)
    
    But this is O(N^2). For N=400, that's 160K operations per gradient, which is fine.
    But we want it fast. Let me use FFT.
    
    Actually, the cleanest way: 
    The autoconvolution ac = h * h (linear). The gradient of sum_m g[m]*ac[m] w.r.t. h[j] is:
    = sum_m g[m] * (h[m-j] + h[j-m])  [with h[k]=0 for k outside [0,N-1]]
    = 2 * sum_m g[m] * h[m-j]  [by symmetry, if g is symmetric... but it may not be]
    
    Hmm, g[m] = weights[m] * sign(ac[m]), and ac is symmetric (ac[m]=ac[size-1-m] for linear autoconv... 
    actually no, linear autoconv of a real sequence is symmetric: ac[m] = ac[size-1-m]).
    
    So g[m] is also symmetric: g[m] = g[size-1-m].
    
    Then: sum_m g[m]*h[m-j] + sum_m g[m]*h[j-m]
    Let k=j-m in second sum: sum_k g[j-k]*h[k] = sum_k g[size-1-(j-k)]*h[k] = sum_k g[size-1-j+k]*h[k]
    
    This is getting messy. Let me just compute it directly - N=400 is small enough.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    
    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)
    
    if weights is None:
        return np.zeros_like(h)
    
    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)
    
    # grad_h[j] = sum_m g_ac[m] * d(ac[m])/d(h[j])
    # d(ac[m])/d(h[j]) = h[m-j] + h[j-m] (with zero padding outside [0,N-1])
    
    # Compute via direct sum (N=400, so O(N^2) = 160K ops, fast)
    grad_h = np.zeros(N)
    for j in range(N):
        # h[m-j] for m in [0, size-1], valid when 0 <= m-j < N
        # h[j-m] for m in [0, size-1], valid when 0 <= j-m < N
        # m-j in [0,N-1] => m in [j, j+N-1]
        # j-m in [0,N-1] => m in [j-N+1, j]
        
        # First term: sum_m g_ac[m] * h[m-j] for m in [j, min(j+N-1, size-1)]
        m_start1 = j
        m_end1 = min(j + N, len(g_ac))
        if m_end1 > m_start1:
            grad_h[j] += np.sum(g_ac[m_start1:m_end1] * h[:m_end1-m_start1])
        
        # Second term: sum_m g_ac[m] * h[j-m] for m in [max(0, j-N+1), min(j, size-1)]
        m_start2 = max(0, j - N + 1)
        m_end2 = min(j + 1, len(g_ac))
        if m_end2 > m_start2:
            # h[j-m] for m in [m_start2, m_end2): indices j-m in [j-m_end2+1, j-m_start2]
            idx = j - np.arange(m_start2, m_end2)
            grad_h[j] += np.sum(g_ac[m_start2:m_end2] * h[idx])
    
    # Now divide by s^2 and subtract the term from derivative of 1/s^2
    # f = sm / s^2
    # df/dh = grad_sm/s^2 - 2*sm/s^3 * ds/dh
    # ds/dh = 1 for each element
    
    df = grad_h / (s * s) - 2 * sm / (s * s * s) * np.ones(N)
    
    return df

def _gradient_descent(h, beta=30.0, lr=0.01, max_iter=2000, deadline=None, tol=1e-8):
    """
    Gradient descent on the smooth surrogate with adaptive learning rate.
    """
    h = h.copy()
    current_val = _smooth_objective(h, beta)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _smooth_gradient(h, beta)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Adaptive learning rate
        step = lr / (1.0 + it * 0.01)
        h_new = h - step * grad
        
        new_val = _smooth_objective(h_new, beta)
        
        if new_val < current_val:
            h = h_new
            current_val = new_val
        else:
            # Reduce step size
            lr *= 0.5
            step = lr / (1.0 + it * 0.01)
            h_new = h - step * grad
            new_val = _smooth_objective(h_new, beta)
            if new_val < current_val:
                h = h_new
                current_val = new_val
            else:
                break
    
    return h, current_val

def _lbfgs_optimize(h, beta=30.0, max_iter=500, deadline=None):
    """
    L-BFGS optimization of the smooth surrogate.
    """
    h0 = h.copy()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)
    
    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'maxfun': 2000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)

def _block_sign_flip_local_search(h, block_sizes=None, max_passes=5, deadline=None):
    """
    Local search: try flipping signs of blocks of consecutive elements.
    O(N log N) per evaluation via FFT.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            
            for start in range(0, N, bs):
                if deadline is not None and time.time() > deadline:
                    break
                
                end = min(start + bs, N)
                h[start:end] *= -1
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def _single_flip_local_search(h, max_passes=10, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    """
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h[i] *= -1
            new_val = _raw_value(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_random(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_window(n, window_type='gaussian'):
    x = np.arange(n)
    if window_type == 'gaussian':
        mu = n/2
        sigma = n/4
        return np.exp(-(x - mu)**2 / (2*sigma**2))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n-1))
    elif window_type == 'triangle':
        mu = n/2
        return np.maximum(0, 1 - np.abs(x - mu)/mu)
    elif window_type == 'blackman':
        a0, a1, a2 = 0.42, 0.5, 0.08
        return a0 - a1*np.cos(2*np.pi*x/(n-1)) + a2*np.cos(4*np.pi*x/(n-1))
    return np.ones(n)

def _gen_smoothed_random(n, seed, kernel_size=15):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode='same')
    return smooth

def _normalize_sum(h, target=1.0):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h
    return h * (target / s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _raw_value(best_h)
    best_score = base_score
    
    # Generate diverse initializations
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N) / N)
    
    # 2. Chirp
    h_chirp = _gen_chirp(N)
    inits.append(_normalize_sum(h_chirp))
    
    # 3. Random sequences
    for seed in [1, 7, 13, 42, 99]:
        h_rand = _gen_random(N, seed)
        inits.append(_normalize_sum(h_rand))
    
    # 4. Windowed
    for wtype in ['gaussian', 'hamming', 'triangle', 'blackman']:
        h_win = _gen_window(N, wtype)
        inits.append(_normalize_sum(h_win))
    
    # 5. Smoothed random
    for seed in [5, 11, 23, 37, 53]:
        h_smooth = _gen_smoothed_random(N, seed)
        inits.append(_normalize_sum(h_smooth))
    
    # 6. Alternating signs with envelope
    for seed in [3, 17]:
        rng = np.random.default_rng(seed)
        env = np.exp(-((np.arange(N) - N/2)**2) / (2*(N/4)**2))
        signs = rng.choice([-1, 1], size=N)
        h_alt = signs * env
        inits.append(_normalize_sum(h_alt))
    
    # 7. Multi-frequency
    for freq_combo in [(1, 2), (1, 3), (2, 5), (3, 7)]:
        f1, f2 = freq_combo
        i = np.arange(N)
        h_multi = np.cos(2*np.pi*f1*i/N) + 0.5*np.cos(2*np.pi*f2*i/N)
        inits.append(_normalize_sum(h_multi))
    
    # Optimization: for each init, try gradient-based then local search
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        h_work = h_init.copy()
        
        # Stage 1: Smooth gradient optimization (L-BFGS or GD)
        remaining = deadline - time.time()
        if remaining > 15:
            h_opt, val_opt = _lbfgs_optimize(h_work, beta=30.0, max_iter=200, 
                                              deadline=deadline - 15)
            h_work = h_opt
        
        # Stage 2: Gradient descent refinement
        remaining = deadline - time.time()
        if remaining > 10:
            h_gd, val_gd = _gradient_descent(h_work, beta=50.0, lr=0.005, 
                                             max_iter=500, deadline=deadline - 10)
            h_work = h_gd
        
        # Stage 3: Block sign-flip local search
        remaining = deadline - time.time()
        if remaining > 8:
            h_block, val_block = _block_sign_flip_local_search(h_work, 
                                                               block_sizes=[1, 2, 4, 8, 16],
                                                               max_passes=3,
                                                               deadline=deadline - 8)
            h_work = h_block
        
        # Stage 4: Single flip local search
        remaining = deadline - time.time()
        if remaining > 5:
            h_flip, val_flip = _single_flip_local_search(h_work, max_passes=5,
                                                         deadline=deadline - 5)
            h_work = h_flip
        
        # Evaluate
        val = _raw_value(h_work)
        if val < best_score:
            best_score = val
            best_h = h_work.copy()
    
    # Final refinement on the best solution
    remaining = deadline - time.time()
    if remaining > 15:
        h_final, val_final = _block_sign_flip_local_search(best_h,
                                                           block_sizes=[1, 2, 4, 8, 16, 32, 64],
                                                           max_passes=5,
                                                           deadline=deadline - 10)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final
    
    remaining = deadline - time.time()
    if remaining > 8:
        h_final2, val_final2 = _single_flip_local_search(best_h, max_passes=10,
                                                         deadline=deadline - 5)
        if val_final2 < best_score:
            best_score = val_final2
            best_h = h_final2
    
    # Ensure we return something valid
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        best_score = _raw_value(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
