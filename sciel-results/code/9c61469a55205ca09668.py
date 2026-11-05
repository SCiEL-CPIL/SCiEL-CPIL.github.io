import numpy as np
import time

N = 400
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _compute_c3_bound(h):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function on [-1/4, 1/4] with N equal intervals:
    dx = 0.5/N
    The autoconvolution is piecewise linear, max |f*f| occurs at grid points.
    C3 bound = dx * max_k |sum_i h_i * h_{i+k}|
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Compute linear autoconvolution via FFT
    # conv[k] = sum_i h[i] * h[i+k] for k = -(n-1), ..., n-1
    # Using FFT: autoconv = ifft(fft(h, 2n) * conj(fft(h, 2n)))
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.fft(h, n=fft_size)
    conv = np.fft.ifft(H * np.conj(H)).real[:n]
    # conv[k] for k=0..n-1 gives sum_i h[i]*h[i+k] (circular, but zero-padded so linear)
    # Actually for autoconvolution we want:
    # acf[k] = sum_{i=0}^{n-1-k} h[i] * h[i+k] for k >= 0
    # This is the linear autocorrelation. Using the FFT approach with zero padding:
    # ifft(|fft(h, 2n)|^2) gives the circular autocorrelation of length 2n
    # The first n elements give the linear autocorrelation
    
    max_abs = np.max(np.abs(conv))
    return dx * max_abs

def _score(h):
    """Compute the C3 upper bound (lower is better)."""
    return _compute_c3_bound(h)

def _compute_acf_linear(h):
    """Compute linear autocorrelation via FFT. Returns array of length n."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.fft(h, n=fft_size)
    acf_full = np.fft.ifft(H * np.conj(H)).real
    # acf_full[k] for k=0..n-1 gives sum_{i} h[i]*h[i+k] (linear, zero-padded)
    return acf_full[:n]

def _psl_value(acf, n):
    """Get PSL value: max abs at non-zero lags."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _compute_neighbor_sums(h):
    """
    Compute for each position i and lag k: h[i-k] + h[i+k] for valid indices.
    Returns array of shape (n, n-1).
    """
    n = len(h)
    neighbor_sums = np.zeros((n, n-1))
    for k in range(1, n):
        neighbor_sums[k:, k-1] += h[:n-k]
        neighbor_sums[:n-k, k-1] += h[k:]
    return neighbor_sums

def _sa_optimize(h, time_budget=10.0, start_time=None, T0=1.0, T_end=1e-8, cooling_rate=0.9995):
    """
    Simulated Annealing optimization with incremental ACF updates.
    Uses the C3 bound as the objective.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    dx = 0.5 / n
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf_linear(h)
    # C3 bound = dx * max(|acf[0]|, max|acf[1:n]|)
    # But acf[0] = sum h[i]^2 = n (for +/-1), so it's fixed
    # The variable part is max|acf[1:n]|
    # Actually, the max over ALL lags including 0 matters for the bound
    # But since acf[0] = n is constant, we only need to track max|acf[1:n]|
    # unless max|acf[1:n]| > n, which it won't for +/-1 sequences
    
    best_psl = _psl_value(acf, n)
    current_psl = best_psl
    neighbor_sums = _compute_neighbor_sums(h)
    
    T = T0
    rng = np.random.default_rng(42)
    
    while (time.time() - start_time) < time_budget and T > T_end:
        # Pick a random index to flip
        i = rng.integers(0, n)
        
        # Calculate change in ACF if we flip h[i]
        # delta_acf[k] = -2 * h[i] * (h[i-k] + h[i+k]) for valid k
        old_val = h[i]
        delta_acf = -2.0 * old_val * neighbor_sums[i, :]
        
        # New ACF lags 1..n-1
        new_acf_lags = acf[1:n] + delta_acf
        new_psl = np.max(np.abs(new_acf_lags))
        
        # Acceptance criterion
        delta = new_psl - current_psl
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept the move
            h[i] = -old_val
            new_val = h[i]
            
            # Update ACF
            acf[1:n] += delta_acf
            
            # Update neighbor sums
            for k in range(1, n):
                if i + k < n:
                    neighbor_sums[i+k, k-1] += 2.0 * new_val
                if i - k >= 0:
                    neighbor_sums[i-k, k-1] += 2.0 * new_val
            
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling_rate
        
        if best_psl < 1e-10:
            break
            
    return best_h, best_psl

def _logistic_init(r, n, seed, init_x=None):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    x_{k+1} = r * x_k * (1 - x_k)
    sign is determined by whether x_k > 0.5
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

def _thue_morse_init(n):
    """Generate Thue-Morse sequence mapped to +/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    signs = np.zeros(n)
    for i in range(n):
        # Count bits
        bits = bin(i).count('1')
        signs[i] = 1.0 if bits % 2 == 0 else -1.0
    return signs

def _quasi_periodic_init(n, seed):
    """Generate quasi-periodic sequence using irrational frequencies."""
    rng = np.random.default_rng(seed)
    # Use golden ratio conjugate for incommensurate frequency
    phi = (1 + np.sqrt(5)) / 2.0
    freq = 1.0 / phi  # incommensurate with 1
    t = np.arange(n)
    # Sum of a few incommensurate frequencies
    val = np.zeros(n)
    for k in range(1, 6):
        freq_k = freq * k + rng.uniform(-0.01, 0.01)
        phase = rng.uniform(0, 2 * np.pi)
        val += np.cos(2 * np.pi * freq_k * t + phase)
    return np.sign(val)

def construct():
    start_time = time.time()
    deadline = TIME_LIMIT
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random sequences
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Thue-Morse (known low autocorrelation)
    candidates.append(_thue_morse_init(N))
    
    # 3. Quasi-periodic with incommensurate frequencies
    for seed in [1, 2, 3, 4, 5]:
        candidates.append(_quasi_periodic_init(N, seed))
    
    # 4. Logistic map initializations - search over parameters
    for trial in range(20):
        r = rng.uniform(3.5, 4.0)
        init_x = rng.uniform(0.1, 0.9)
        h = _logistic_init(r, N, 42 + trial, init_x)
        candidates.append(h)
    
    # 5. Simple periodic patterns (as baselines)
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
    
    # 6. Block structure
    for block_size in [5, 10, 20]:
        for i in range(3):
            h = np.zeros(N)
            n_blocks = N // block_size
            for b in range(n_blocks):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            h[N - N % block_size:] = rng.choice([-1.0, 1.0], size=N % block_size)
            candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with SA
    top_k = 8
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time()
        candidates_left = top_k - i
        time_per_candidate = max(2.0, (time_left - 10.0) / max(1, candidates_left))
        
        h_refined, s_refined_psl = _sa_optimize(h, time_budget=time_per_candidate, start_time=start_time)
        s_refined = _score(h_refined)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do random restarts with SA
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        
        time_left = deadline - time.time()
        time_per_candidate = max(2.0, min(15.0, (time_left - 5.0) / 5.0))
        
        h_refined, s_refined_psl = _sa_optimize(h_init, time_budget=time_per_candidate, start_time=start_time)
        s_refined = _score(h_refined)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final local search polish on the best solution
    if best_h is not None and time.time() < deadline - 5:
        # Greedy local search: try flipping each bit and keep if it improves
        h = best_h.copy()
        current_score = _score(h)
        improved = True
        max_sweeps = 5
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline - 3:
            improved = False
            sweep += 1
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                old_val = h[i]
                h[i] = -old_val
                new_score = _score(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_h = h
    
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
