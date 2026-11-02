import numpy as np
import time

N = 50
_TIME_LIMIT = 140.0
_start_time = time.time()

def _ratio(h):
    """Compute the objective ratio."""
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _ratio_with_p(h, p):
    """
    Compute the expected autoconvolution peak under attacker distribution p.
    This is the defender's cost in the min-max game.
    """
    conv = np.convolve(h, h, mode='full')
    expected_peak = np.dot(p, conv)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12 or expected_peak < 1e-12:
        return 0.0
    return s2 / (s1 * expected_peak)

def _game_solve(h0, max_iter=300, eta_h=0.3, eta_p=0.5, tol=1e-10):
    """
    Solve the min-max game using multiplicative (mirror) updates.
    
    Defender (h): multiplicative update to minimize expected peak
    Attacker (p): multiplicative update to maximize expected peak
    
    At saddle point: ratio = ||h||_2^2 / (||h||_1 * max_k conv_k)
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 1e-10, None)
    h = h / np.sum(h)  # normalize to sum=1
    
    # Attacker distribution over lags 0..2N-2
    n_lags = 2 * N - 1
    p = np.ones(n_lags) / n_lags
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        
        conv = np.convolve(h, h, mode='full')
        s2 = np.dot(h, h)
        s1 = np.sum(h)
        
        # Attacker update: increase p_k proportional to conv_k
        # p_new_k = p_k * conv_k / sum(p * conv)
        expected_peak = np.dot(p, conv)
        if expected_peak < 1e-12:
            break
        
        p_new = p * conv
        p_new_sum = np.sum(p_new)
        if p_new_sum < 1e-12:
            break
        p_new = p_new / p_new_sum
        
        # Defender update: reduce h_i where it contributes most to expected peak
        # d/dh_i of expected_peak = sum_k p_k * (h_{k-i} + h_{i-k})
        # We want to minimize: expected_peak / s2 (since ratio = s2/expected_peak)
        # Gradient of expected_peak w.r.t. h_i:
        d_peak_dh = np.zeros(N)
        for i in range(N):
            for k in range(n_lags):
                if p[k] < 1e-12:
                    continue
                j1 = k - i
                j2 = i - k
                term = 0.0
                if 0 <= j1 < N:
                    term += h[j1]
                if 0 <= j2 < N:
                    term += h[j2]
                d_peak_dh[i] += p[k] * term
        
        # Multiplicative update: h_new_i = h_i * exp(-eta * d_peak_dh_i / expected_peak)
        # This reduces h_i where it contributes most to the peak
        h_new = h * np.exp(-eta_h * d_peak_dh / expected_peak)
        h_new = np.clip(h_new, 1e-10, None)
        h_new = h_new / np.sum(h_new)
        
        # Compute score with current p
        score_p = _ratio_with_p(h_new, p_new)
        
        # Also compute the true max-convolution ratio
        score_true = _ratio(h_new)
        
        if score_true > best_score:
            best_score = score_true
            best_h = h_new.copy()
        
        # Update
        h = h_new
        p = p_new
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol and np.max(np.abs(p_new - p)) < tol:
            break
    
    return best_h, best_score

def _game_solve_fast(h0, max_iter=500, eta_h=0.5, eta_p=1.0):
    """
    Faster game solver using vectorized operations.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 1e-10, None)
    h = h / np.sum(h)
    
    n_lags = 2 * N - 1
    p = np.ones(n_lags) / n_lags
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        
        conv = np.convolve(h, h, mode='full')
        s2 = np.dot(h, h)
        
        # Attacker update
        expected_peak = np.dot(p, conv)
        if expected_peak < 1e-12:
            break
        p = p * conv
        p = p / np.sum(p)
        
        # Defender update (vectorized)
        # d_peak_dh[i] = sum_k p[k] * (h[k-i] + h[i-k])
        # This is a correlation of p with h, evaluated at each shift
        # d_peak_dh[i] = (p * h)[i] where * is cross-correlation
        # Actually: d_peak_dh[i] = sum_k p[k] h[k-i] + sum_k p[k] h[i-k]
        #           = sum_k p[k] h[k-i] + sum_k p[k-i] h[k]  (by substitution)
        #           = (p ⊛ h)[i] + (h ⊛ p)[i]  where ⊛ is cross-correlation
        
        # Compute using FFT for speed
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        
        H = np.fft.rfft(h, n_fft)
        P = np.fft.rfft(p, n_fft)
        
        # Cross-correlation: (p ⊛ h)[i] = sum_k p[k] h[k-i]
        # In FFT: ifft(fft(p) * conj(fft(h)))
        cross1 = np.fft.irfft(P * np.conj(H), n_fft)[:N]
        cross2 = np.fft.irfft(H * np.conj(P), n_fft)[:N]
        
        d_peak_dh = cross1 + cross2
        d_peak_dh = np.clip(d_peak_dh, 0, None)
        
        # Multiplicative update
        h_new = h * np.exp(-eta_h * d_peak_dh / expected_peak)
        h_new = np.clip(h_new, 1e-10, None)
        h_new = h_new / np.sum(h_new)
        
        score_true = _ratio(h_new)
        if score_true > best_score:
            best_score = score_true
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_score

def _spectral_init():
    """Initialize with a smooth window that has low sidelobes."""
    # Use a raised cosine / Hann window
    t = np.linspace(0, 1, N)
    h = 0.5 * (1 + np.cos(2 * np.pi * t))
    h = np.clip(h, 0, None)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _gaussian_init(sigma):
    """Gaussian-shaped initialization."""
    center = (N - 1) / 2.0
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    h = np.clip(h, 0, None)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _bimodal_init(gap, weight):
    """Two concentrated peaks."""
    h = np.zeros(N)
    h[0] = weight
    h[gap] = 1.0 - weight
    return h

def construct():
    deadline = _start_time + _TIME_LIMIT - 10
    best_h = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # Single spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    inits.append(h_spike)
    
    # Two-point concentrations at various gaps
    for gap in [1, 2, 3, 5, 7, 10, 13, 17, 25]:
        for w in [0.3, 0.5, 0.7]:
            inits.append(_bimodal_init(gap, w))
    
    # Smooth windows
    inits.append(_spectral_init())
    for sigma in [2, 3, 4, 5, 7, 10]:
        inits.append(_gaussian_init(sigma))
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(8):
        h = rng.random(N)
        h = h / np.sum(h)
        inits.append(h)
    
    # Sparse random (few nonzeros)
    for _ in range(5):
        h = np.zeros(N)
        n_nonzero = rng.integers(3, 15)
        idx = rng.choice(N, size=n_nonzero, replace=False)
        h[idx] = rng.random(n_nonzero)
        h = h / np.sum(h)
        inits.append(h)
    
    # Run game solver on each initialization
    for h0 in inits:
        if time.time() > deadline:
            break
        
        # Fast game solver
        h_opt, s_opt = _game_solve_fast(h0, max_iter=200, eta_h=0.4)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # If time permits, refine with slower solver
        if time.time() < deadline - 20:
            h_opt2, s_opt2 = _game_solve(h_opt, max_iter=100, eta_h=0.2, eta_p=0.3)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()
    
    # Final polish: run game solver from best with small steps
    if best_h is not None and time.time() < deadline - 5:
        h_final, s_final = _game_solve_fast(best_h, max_iter=100, eta_h=0.1)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return [float(x) for x in best_h]
