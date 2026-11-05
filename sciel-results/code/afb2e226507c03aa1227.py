import numpy as np
import time
import math

N = 400

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

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    acf = _compute_acf(h)
    return _psl_value(acf, N)

def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    # Pre-compute the current ACF lags
    acf_lags = acf[1:n]
    
    # To speed this up, we can vectorize the computation of the new ACF for a given i.
    # new_acf = acf + delta_i
    # delta_i[k-1] = -2 * h[i] * (h[i-k] + h[i+k])
    
    # Let's try a slightly more optimized loop or vectorization.
    # For each i, the change in ACF is:
    # d_acf = -2 * h[i] * (convolve-like term)
    
    # Actually, let's just loop over i, but make it tight.
    # N=400, so 400 iterations. Inside, we do vector ops of length 399.
    # This should be fast enough.
    
    for i in range(n):
        # Construct the change in ACF
        # h[i-k] is valid for k <= i
        # h[i+k] is valid for k < n-i
        # So the neighbor sum vector:
        neighbor_sum = np.zeros(n-1)
        
        # h[i-k] for k=1..min(i, n-1)
        # indices: i-1, i-2, ..., max(0, i-(n-1))
        # corresponding k: 1, 2, ..., min(i, n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1] # h[i-1], h[i-2]...
            
        # h[i+k] for k=1..min(n-1-i, n-1)
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1] # h[i+1], h[i+2]...
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _simulated_annealing_psl(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using the fast gradient to evaluate moves.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        # Compute delta for this specific i
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl

def _generate_fbmbased(hurst, rng):
    """
    Generate a signed step function based on a deterministic fractional Brownian motion
    with a given Hurst exponent.
    """
    n = N
    # Generate white noise
    z = rng.standard_normal(n)
    
    # Compute FFT of white noise
    Z = np.fft.rfft(z)
    
    # Apply fractional Brownian motion filter
    # The power spectral density of fBm is proportional to f^(-2*H - 1)
    # So the amplitude spectrum is proportional to f^(-H - 0.5)
    
    # Frequencies
    freqs = np.fft.rfftfreq(n)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    
    # Filter coefficients
    # We want the spectrum to decay like f^(-H - 0.5)
    # So we multiply the white noise spectrum by f^(-H - 0.5)
    # But we also need to ensure the output is real, so we handle the FFT carefully
    
    # For a real signal, we can just apply the filter in the frequency domain
    # and take the inverse FFT
    filter_coeff = np.power(freqs, -hurst - 0.5)
    Z_filtered = Z * filter_coeff
    
    # Inverse FFT
    x = np.fft.irfft(Z_filtered, n=n)
    
    # Normalize and take sign
    x = x - np.mean(x)
    x = x / np.std(x)
    
    # Take sign to get +/-1
    return np.sign(x)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
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
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # 5. Fractional Brownian Motion based candidates
    # Coarse grid search over Hurst exponent
    hurst_values = np.linspace(0.1, 0.9, 10)
    for hurst in hurst_values:
        candidates.append(_generate_fbmbased(hurst, rng))
        
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates with fast PSL optimization (Gordon's method)
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 20:
            break
        # Allocate time based on rank
        time_limit = deadline - 10
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do a few random restarts with fast PSL optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick optimization with limited time
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=100, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If we still have time, try SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
