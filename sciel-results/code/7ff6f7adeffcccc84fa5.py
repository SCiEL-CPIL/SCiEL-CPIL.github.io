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

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    h = np.zeros(n)
    for i in range(n):
        # Count number of 1s in binary representation
        count = 0
        x = i
        while x > 0:
            count += x & 1
            x >>= 1
        h[i] = 1.0 if count % 2 == 0 else -1.0
    return h

def _fast_flip_score(h, H, j, fft_size):
    """
    Compute new score after flipping h[j] using FFT update.
    H is the current rfft of h.
    """
    n = len(h)
    # Update H: H_new = H - 2*h[j]*exp(-2*pi*i*j*freq)
    freqs = np.fft.rfftfreq(fft_size) * fft_size
    phase = -2j * np.pi * j * freqs / fft_size
    H_new = H - 2 * h[j] * np.exp(phase)
    
    # Compute new ACF
    acf_new = np.fft.irfft(np.abs(H_new) ** 2, n=fft_size)
    return np.max(np.abs(acf_new[1:n]))

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Helper: Fast local search using greedy flipping with FFT updates
    def _local_search(h, time_limit):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        # Precompute H
        H = np.fft.rfft(h, n=fft_size)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_pos = -1
            best_new_score = current_score
            
            for j in range(n):
                new_score = _fast_flip_score(h, H, j, fft_size)
                if new_score < best_new_score - 1e-9:
                    best_new_score = new_score
                    best_pos = j
            
            if best_pos != -1:
                h[best_pos] = -h[best_pos]
                # Update H
                freqs = np.fft.rfftfreq(fft_size) * fft_size
                phase = -2j * np.pi * best_pos * freqs / fft_size
                H = H - 2 * (-h[best_pos]) * np.exp(phase)  # h[best_pos] was flipped, so old value was -h[best_pos]
                current_score = best_new_score
                improved = True
                
        return h, current_score

    # Helper: Simulated annealing
    def _simulated_annealing(h, time_limit):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        H = np.fft.rfft(h, n=fft_size)
        
        T = 1.0
        cooling = 0.999
        rng_sa = np.random.default_rng(42)
        
        while time.time() < time_limit and T > 0.001:
            j = rng_sa.integers(0, n)
            old_val = h[j]
            h[j] = -h[j]
            new_score = _fast_flip_score(h, H, j, fft_size)
            
            delta = new_score - current_score
            if delta < 0 or rng_sa.random() < np.exp(-delta / T):
                # Accept
                # Update H
                freqs = np.fft.rfftfreq(fft_size) * fft_size
                phase = -2j * np.pi * j * freqs / fft_size
                H = H - 2 * old_val * np.exp(phase)
                current_score = new_score
            else:
                # Reject
                h[j] = old_val
            
            T *= cooling
            
        return h, current_score

    # Generate initial candidates
    candidates = []
    
    # 1. Random
    rng = np.random.default_rng(42)
    for i in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Thue-Morse sequences (known low autocorrelation)
    candidates.append(_thue_morse(N))
    # Shifted Thue-Morse
    tm = _thue_morse(N)
    candidates.append(np.roll(tm, 1))
    candidates.append(np.roll(tm, 2))
    
    # 3. Quasi-periodic (Irrational frequencies)
    for trial in range(4):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 4. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 5. Block structure
    block_size = 10
    for i in range(3):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)

    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Phase 1: Refine top candidates with local search
    top_k = min(5, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 60:
            break
        s, h = scored_candidates[i]
        time_limit = min(deadline - 30, time.time() + 10)
        h_refined, s_refined = _local_search(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Phase 2: Simulated annealing on best candidate
    if best_h is not None and time.time() < deadline - 30:
        time_limit = deadline - 20
        h_sa, s_sa = _simulated_annealing(best_h, time_limit)
        if s_sa < best_score:
            best_score = s_sa
            best_h = h_sa.copy()
    
    # Phase 3: Random restarts with quick local search
    while time.time() < deadline - 15:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 10, time.time() + 3)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Phase 4: Final polish on best candidate
    if best_h is not None and time.time() < deadline - 5:
        time_limit = deadline - 2
        h_refined, s_refined = _local_search(best_h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
