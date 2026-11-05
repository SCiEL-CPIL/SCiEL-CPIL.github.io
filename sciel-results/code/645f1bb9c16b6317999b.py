import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _compute_neighbor_sums(h):
    """
    Compute for each position i the vector of neighbor sums:
    neighbor_sum[i][k-1] = h[i-k] + h[i+k] for valid k.
    Returns array of shape (n, n-1).
    """
    n = len(h)
    neighbor_sums = np.zeros((n, n-1))
    for k in range(1, n):
        neighbor_sums[k:, k-1] += h[:n-k]
        neighbor_sums[:n-k, k-1] += h[k:]
    return neighbor_sums

def _softmax_psl(acf_lags, beta=50.0):
    """
    Smooth surrogate for max(|acf_lags|) using log-sum-exp.
    As beta -> inf, this approaches the hard max.
    """
    x = np.abs(acf_lags)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def _sa_psl_optimize(h, time_budget=10.0, start_time=None, T0=1.0, T_end=1e-6, cooling_rate=0.999):
    """
    Simulated Annealing PSL optimization with incremental ACF updates.
    Uses soft-max surrogate for smoother acceptance landscape.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    current_psl = best_psl
    neighbor_sums = _compute_neighbor_sums(h)
    
    T = T0
    rng = np.random.default_rng(42)
    
    beta_start = 10.0
    beta_end = 200.0
    
    step = 0
    total_steps_estimate = max(1, int(time_budget * 5000))
    
    while (time.time() - start_time) < time_budget and T > T_end:
        progress = min(1.0, step / total_steps_estimate)
        beta = beta_start + (beta_end - beta_start) * progress
        
        i = rng.integers(0, n)
        
        old_val = h[i]
        delta_acf = -2.0 * old_val * neighbor_sums[i, :]
        
        new_acf_lags = acf[1:n] + delta_acf
        
        current_soft = _softmax_psl(acf[1:n], beta)
        new_soft = _softmax_psl(new_acf_lags, beta)
        
        delta = new_soft - current_soft
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h[i] = -old_val
            new_val = h[i]
            
            acf[1:n] += delta_acf
            
            for k in range(1, n):
                if i + k < n:
                    neighbor_sums[i+k, k-1] += 2.0 * new_val
                if i - k >= 0:
                    neighbor_sums[i-k, k-1] += 2.0 * new_val
            
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling_rate
        step += 1
        
        if best_psl < 1e-10:
            break
            
    return best_h, best_psl

def _polish_with_softmax(h, time_budget=5.0, start_time=None):
    """
    Final polish pass: use soft-max to find sub-peak reductions,
    then verify with hard max.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    best_h = h.copy()
    neighbor_sums = _compute_neighbor_sums(h)
    
    rng = np.random.default_rng(123)
    beta = 100.0
    T = 0.5
    
    while (time.time() - start_time) < time_budget:
        i = rng.integers(0, n)
        
        old_val = h[i]
        delta_acf = -2.0 * old_val * neighbor_sums[i, :]
        new_acf_lags = acf[1:n] + delta_acf
        
        current_soft = _softmax_psl(acf[1:n], beta)
        new_soft = _softmax_psl(new_acf_lags, beta)
        
        delta = new_soft - current_soft
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h[i] = -old_val
            new_val = h[i]
            acf[1:n] += delta_acf
            
            for k in range(1, n):
                if i + k < n:
                    neighbor_sums[i+k, k-1] += 2.0 * new_val
                if i - k >= 0:
                    neighbor_sums[i-k, k-1] += 2.0 * new_val
            
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= 0.9995
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _fbm_signal(n, H, seed):
    """
    Generate a fractional Brownian motion signal of length n with Hurst exponent H.
    Uses the spectral synthesis method: generate white noise in frequency domain,
    multiply by fBm power spectrum S(f) ~ |f|^{-(2H+1)}, then inverse FFT.
    
    H in (0, 1):
      H = 0.5: white noise (no correlation)
      H > 0.5: long-range positive correlation (smooth)
      H < 0.5: anti-correlation (rough)
    """
    rng = np.random.default_rng(seed)
    
    # Generate white noise in frequency domain
    freqs = np.fft.rfftfreq(n)
    
    # Power spectrum of fBm: S(f) ~ |f|^{-(2H+1)}
    # Avoid zero frequency
    mask = freqs != 0
    spectrum = np.zeros(len(freqs))
    spectrum[mask] = np.abs(freqs[mask]) ** (-(2*H + 1))
    spectrum[0] = 0  # Remove DC component
    
    # Generate complex white noise
    noise_real = rng.standard_normal(len(freqs))
    noise_imag = rng.standard_normal(len(freqs))
    noise = noise_real + 1j * noise_imag
    
    # Multiply by spectrum
    colored = spectrum * noise
    
    # Inverse FFT to get time-domain fBm signal
    signal = np.fft.irfft(colored, n=n)
    
    # Remove mean
    signal -= np.mean(signal)
    
    # Normalize
    std = np.std(signal)
    if std > 1e-15:
        signal /= std
    
    return signal

def _fbm_sign_sequence(n, H, seed, threshold=0.0):
    """
    Generate a signed step function from fBm signal.
    h[i] = sign(signal[i] - threshold)
    """
    signal = _fbm_signal(n, H, seed)
    signs = np.sign(signal - threshold)
    signs[signs == 0] = 1.0  # Replace zeros with +1
    return signs

def _optimize_fbm_params(n, seed, max_trials=30):
    """
    Search over fBm Hurst exponent H and seed to find sequences with low PSL.
    Returns the best (h, psl) pair.
    """
    rng = np.random.default_rng(seed)
    best_h = None
    best_psl = float('inf')
    
    # Coarse grid search over H in (0.1, 0.9)
    H_vals = np.linspace(0.1, 0.9, 15)
    
    for H in H_vals:
        for trial in range(max_trials // len(H_vals) + 1):
            s = seed + int(H * 1000) + trial
            h = _fbm_sign_sequence(n, H, s)
            psl = _score(h)
            
            if psl < best_psl:
                best_psl = psl
                best_h = h.copy()
    
    # Fine search around best H
    if best_h is not None and len(H_vals) > 1:
        best_H_idx = np.argmin([_score(_fbm_sign_sequence(n, H, seed + int(H * 1000))) for H in H_vals])
        best_H = H_vals[best_H_idx]
        
        # Refine around best H
        H_fine = np.linspace(max(0.05, best_H - 0.1), min(0.95, best_H + 0.1), 10)
        for H in H_fine:
            s = seed + int(H * 10000)
            h = _fbm_sign_sequence(n, H, s)
            psl = _score(h)
            
            if psl < best_psl:
                best_psl = psl
                best_h = h.copy()
    
    return best_h, best_psl

def _logistic_init(r, n, seed, init_x=None):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    """
    rng = np.random.default_rng(seed)
    if init_x is None:
        x = rng.uniform(0.1, 0.9)
    else:
        x = init_x
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            x = rng.uniform(0.1, 0.9)
    return signs

def _optimize_logistic_params(n, seed, max_trials=30):
    """
    Search over logistic map parameters (r, init_x) to find sequences with low PSL.
    """
    rng = np.random.default_rng(seed)
    best_h = None
    best_psl = float('inf')
    
    for trial in range(max_trials):
        r = rng.uniform(3.5, 4.0)
        init_x = rng.uniform(0.1, 0.9)
        
        h = _logistic_init(r, n, seed + trial, init_x)
        psl = _score(h)
        
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(8):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    block_size = 10
    for i in range(8):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # 5. fBm-based initializations with Hurst exponent search (NEW STRUCTURAL IDEA)
    h_fbm, psl_fbm = _optimize_fbm_params(N, seed=42, max_trials=25)
    if h_fbm is not None:
        candidates.append(h_fbm)
    
    # Additional fBm with different seeds
    for seed in [100, 200, 300]:
        h_fbm, psl_fbm = _optimize_fbm_params(N, seed=seed, max_trials=15)
        if h_fbm is not None:
            candidates.append(h_fbm)
    
    # 6. Logistic map initializations
    h_log, psl_log = _optimize_logistic_params(N, seed=42, max_trials=20)
    if h_log is not None:
        candidates.append(h_log)
    
    for seed in [100, 200]:
        h_log, psl_log = _optimize_logistic_params(N, seed=seed, max_trials=10)
        if h_log is not None:
            candidates.append(h_log)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with SA optimization (soft-max surrogate)
    top_k = 10
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 25:
            break
        
        time_left = deadline - time.time()
        candidates_left = top_k - i
        time_per_candidate = max(1.0, time_left / candidates_left)
        
        h_refined, s_refined = _sa_psl_optimize(h, time_budget=time_per_candidate, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do random restarts with SA
    while time.time() < deadline - 12:
        h_init = rng.choice([-1.0, 1.0], size=N)
        
        time_left = deadline - time.time()
        time_per_candidate = max(1.0, time_left / 8.0)
        
        h_refined, s_refined = _sa_psl_optimize(h_init, time_budget=time_per_candidate, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final polish pass with soft-max on the best candidate
    if best_h is not None and time.time() < deadline - 3:
        polish_time = min(6.0, deadline - 3.0 - time.time())
        if polish_time > 1.0:
            h_polished, s_polished = _polish_with_softmax(best_h, time_budget=polish_time, start_time=start_time)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()

    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
