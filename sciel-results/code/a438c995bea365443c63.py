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

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Generate diverse initial candidates using quasi-periodic structures and random restarts.
    # 2. Use a fast local search (greedy flipping) to refine candidates.
    # 3. Use a more sophisticated search (e.g., tabu or simple SA) if time permits, but focus on fast local search 
    #    with multiple restarts as it's often effective for binary optimization.
    
    # Helper: Fast local search using greedy flipping
    def _local_search(h, time_limit):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
            
        # Precompute basis vectors for fast update? 
        # Actually, recomputing FFT for each flip is O(N log N). 
        # N=400, so 400 * 400*log(400) ~ 400 * 400 * 9 = 1.4M ops per pass.
        # A pass takes ~0.01-0.05s. 
        # We can do many passes.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_pos = -1
            best_new_score = current_score
            
            # Compute current ACF once for reference? No, we need to evaluate flips.
            # To speed up, we can use the linearity of the autocorrelation change.
            # ACF(k) = sum h[i]h[i+k].
            # If we flip h[j] to -h[j], the change in ACF(k) is:
            # For lag k:
            #   if j+k in [0, N-1]: term h[j]h[j+k] becomes -h[j]h[j+k] -> change -2 h[j]h[j+k]
            #   if j-k in [0, N-1]: term h[j-k]h[j] becomes -h[j-k]h[j] -> change -2 h[j-k]h[j]
            #   if j==k (lag 0, not considered)
            #   if j==j+k (k=0, not considered)
            #   Note: if j+k == j-k (i.e., k=0), handled.
            #   Also, the term at lag k where i=j and i+k=j+k, and i=j-k and i+k=j.
            #   Wait, the autocorrelation is symmetric. 
            #   Let's just compute the change in the max absolute value.
            #   This is hard to do analytically for the MAX.
            #   So we stick to recomputing FFT, but we can optimize the loop.
            
            # Alternative: Use the fact that flipping h[j] changes the ACF by -2 * h[j] * (h shifted)
            # Let's try to compute the new ACF efficiently.
            # H_new = H - 2*h[j]*exp(-2*pi*i*j*freq)
            # This is complex.
            
            # Let's stick to the simple but robust method: try all flips, keep the best.
            # To speed up, we can vectorize the score calculation if we had a way to update ACF.
            # But for now, let's just do the loop. It's fast enough for N=400.
            
            for j in range(n):
                h[j] = -h[j]
                new_score = _score(h)
                if new_score < best_new_score - 1e-9:
                    best_new_score = new_score
                    best_pos = j
                    best_delta = new_score - current_score
                h[j] = -h[j] # revert
            
            if best_pos != -1:
                h[best_pos] = -h[best_pos]
                current_score = best_new_score
                improved = True
                
        return h, current_score

    # Generate initial candidates
    candidates = []
    
    # 1. Random
    rng = np.random.default_rng(42)
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    # f(t) = sign(sum a_k cos(2*pi*k*t + phi_k))
    # This creates a structured pattern.
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
        
    # 4. Sparse combinatorial design (from knowledge)
    # Try a pattern with few non-zero-like signs? No, must be +/- 1.
    # Maybe a block structure.
    block_size = 10
    for i in range(4):
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
    
    # Refine the top 5 candidates with local search
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 20:
            break
        # Allocate time based on rank
        time_limit = deadline - 10
        h_refined, s_refined = _local_search(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do a few random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick local search with limited time
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
