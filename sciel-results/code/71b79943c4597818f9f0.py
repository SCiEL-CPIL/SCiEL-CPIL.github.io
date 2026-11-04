"""Combined approach: efficient local search with ACF delta updates and diverse initialization."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    return _psl_value(acf, len(h))

def _local_search_fast(h, time_limit):
    """
    Fast local search using ACF delta updates for single flips.
    When flipping h[i] to -h[i], the change in ACF at lag k is:
      delta[k] = -2 * h[i] * (h[i-k] + h[i+k])  (where indices are valid)
    We track the max |ACF| efficiently by updating only affected lags.
    """
    n = len(h)
    h = h.copy()
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    # Precompute valid index ranges for each position
    # For position i, valid k values are 1..n-1 such that i-k >= 0 and i+k < n
    # Actually, ACF is defined for all lags, but we only care about 1..n-1
    # For a flip at position i, the change affects ACF[k] for k where
    # the term h[i]*h[i+k] or h[i-k]*h[i] appears.
    # ACF[k] = sum_{j} h[j]*h[j+k]
    # If we flip h[i], then:
    #   ACF[k] changes by -2*h[i]*h[i+k] (if i+k in [0,n-1])
    #   ACF[k] changes by -2*h[i-k]*h[i] (if i-k in [0,n-1])
    #   Note: ACF[k] = ACF[-k], but we only track k=1..n-1
    
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < time_limit:
        improved = False
        best_i = -1
        best_new_psl = current_psl
        best_delta = 0
        
        # For each position i, compute the new PSL if we flip h[i]
        # To speed up, we can compute the delta ACF for each i and find the max |new ACF|
        # But computing max over all k for each i is O(n^2) per pass.
        # Instead, let's try a heuristic: only consider flips that might reduce the current max.
        
        # Find the lag(s) where the max is achieved
        max_lag_idx = np.argmax(np.abs(acf[1:n]))
        max_lag = max_lag_idx + 1
        max_val = acf[max_lag]
        
        # Try flipping positions that contribute to the max lag
        # For lag k, ACF[k] = sum h[j]*h[j+k]
        # Flipping h[i] changes ACF[k] by -2*h[i]*(h[i+k] + h[i-k])
        # To reduce |ACF[k]|, we want the change to oppose the current sign
        
        # Simple strategy: try all positions, but use a fast approximation
        # For now, let's just try all positions with a quick check
        
        for i in range(n):
            # Compute delta ACF for flipping position i
            # delta[k] = -2 * h[i] * (h[i-k] + h[i+k]) for valid k
            # We only need to check if this reduces the overall max
            
            # Quick check: what is the change in the current max lag?
            k = max_lag
            delta_k = 0
            if i + k < n:
                delta_k -= 2 * h[i] * h[i + k]
            if i - k >= 0:
                delta_k -= 2 * h[i - k] * h[i]
            
            new_max_val = max_val + delta_k
            # This is a lower bound on the new max (other lags might be larger)
            # But it's a quick filter
            
            # For a more thorough check, compute the full new PSL
            # This is expensive, so let's do it only for promising candidates
            if abs(new_max_val) < abs(max_val) - 0.1:  # Promising
                # Compute full new ACF
                h[i] = -h[i]
                new_acf = _compute_acf(h)
                new_psl = _psl_value(new_acf, n)
                h[i] = -h[i]  # revert
                
                if new_psl < best_new_psl - 1e-9:
                    best_new_psl = new_psl
                    best_i = i
                    best_delta = new_psl - current_psl
        
        if best_i != -1:
            h[best_i] = -h[best_i]
            acf = _compute_acf(h)
            current_psl = best_new_psl
            improved = True
            no_improve_count = 0
        else:
            improved = False
    
    return h, current_psl

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random
    for i in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic structures
    for trial in range(10):
        phases = rng.uniform(0, 2 * np.pi, size=8)
        amps = rng.uniform(0.5, 1.5, size=8)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 9):
            val += amps[k-1] * np.cos(2 * np.pi * k * t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23]:
        pattern = np.sign(np.cos(2 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
        
    # 4. Block structures
    for block_size in [5, 10, 20, 25, 40, 50]:
        if N % block_size == 0:
            h = np.zeros(N)
            for b in range(N // block_size):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            candidates.append(h)
    
    # 5. Thue-Morse-like sequences
    def _thue_morse(n):
        tm = np.zeros(n, dtype=float)
        for i in range(n):
            # Count number of 1s in binary representation
            tm[i] = 1.0 if bin(i).count('1') % 2 == 0 else -1.0
        return tm
    
    candidates.append(_thue_morse(N))
    candidates.append(-_thue_morse(N))
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with local search
    top_k = min(8, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 30:
            break
        s, h = scored_candidates[i]
        time_limit = deadline - 15
        h_refined, s_refined = _local_search_fast(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 8)
        h_refined, s_refined = _local_search_fast(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
