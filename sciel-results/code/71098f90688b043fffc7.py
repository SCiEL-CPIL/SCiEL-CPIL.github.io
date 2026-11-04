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
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Helper: Fast local search using greedy flipping with efficient ACF update
    def _local_search(h, time_limit):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
            
        # Precompute the frequency basis for fast updates
        # When we flip h[j], the change in the power spectrum is:
        # H_new = H - 2*h[j]*exp(-2*pi*i*j*k/fft_size) for each frequency k
        # This is a rank-1 update. We can exploit this.
        
        # Actually, let's use a simpler but still efficient approach:
        # Precompute the autocorrelation contributions for each position.
        # ACF[k] = sum_{i} h[i] * h[i+k]
        # If we flip h[j], the new ACF[k] = ACF[k] - 2*h[j]*(h[j+k] + h[j-k]) for valid indices
        # But we need to be careful with boundaries.
        
        # Let's use a vectorized approach to evaluate all flips at once.
        # This is O(N^2) per pass, but with N=400, that's 160k operations, which is fast.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_pos = -1
            best_new_score = current_score
            
            # Compute the current ACF
            H = np.fft.rfft(h, n=fft_size)
            acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
            
            # For each position j, compute the change in the max absolute ACF if we flip h[j]
            # This is expensive to do exactly, so we use a heuristic:
            # Try a subset of positions first, then refine.
            
            # Vectorized flip evaluation:
            # For each j, compute the new ACF by updating the FFT.
            # This is O(N * fft_size) which is too slow.
            
            # Instead, let's use the direct ACF update formula.
            # ACF_new[k] = ACF[k] - 2*h[j]*(h[j+k] + h[j-k]) for k in 1..N-1
            # where h[j+k] is 0 if j+k >= N, and h[j-k] is 0 if j-k < 0.
            
            # Precompute the "influence" of each position on each lag.
            # influence[j, k] = h[j+k] + h[j-k] (with boundary handling)
            
            # This is a N x N matrix. N=400, so 160k elements. Fine.
            influence = np.zeros((n, n-1))
            for j in range(n):
                for k in range(1, n):
                    val = 0.0
                    if j + k < n:
                        val += h[j + k]
                    if j - k >= 0:
                        val += h[j - k]
                    influence[j, k-1] = val
            
            # Now, for each j, the new ACF is:
            # ACF_new[k] = acf[k] - 2*h[j]*influence[j, k]
            # We want to minimize max(|ACF_new[k]|) over k=0..N-2
            
            # This is still expensive to do for all j. Let's use a greedy approach:
            # Try flipping each position and compute the new max.
            
            for j in range(n):
                # Compute new ACF
                new_acf = acf.copy()
                new_acf[1:n] -= 2 * h[j] * influence[j, :]
                new_score = np.max(np.abs(new_acf[1:n]))
                
                if new_score < best_new_score - 1e-9:
                    best_new_score = new_score
                    best_pos = j
                    best_delta = new_score - current_score
            
            if best_pos != -1:
                h[best_pos] = -h[best_pos]
                current_score = best_new_score
                improved = True
                # Recompute influence after flip
                j = best_pos
                for k in range(1, n):
                    val = 0.0
                    if j + k < n:
                        val += h[j + k]
                    if j - k >= 0:
                        val += h[j - k]
                    influence[j, k-1] = val
                # Also, the influence of other positions j' on lag k changes if j' +/- k == j
                # This is complex. Let's just recompute the whole influence matrix.
                # It's O(N^2) = 160k ops, which is fast.
                for jj in range(n):
                    for k in range(1, n):
                        val = 0.0
                        if jj + k < n:
                            val += h[jj + k]
                        if jj - k >= 0:
                            val += h[jj - k]
                        influence[jj, k-1] = val
                
        return h, current_score

    # Helper: Simulated annealing
    def _simulated_annealing(h, time_limit, t_start=1.0, t_end=0.01, alpha=0.999):
        h = h.copy()
        n = len(h)
        current_score = _score(h)
        best_h = h.copy()
        best_score_local = current_score
        
        t = t_start
        rng = np.random.default_rng()
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        while time.time() < time_limit and t > t_end:
            # Random flip
            j = rng.integers(0, n)
            h[j] = -h[j]
            new_score = _score(h)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / t):
                current_score = new_score
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_h = h.copy()
            else:
                h[j] = -h[j]  # revert
            
            t *= alpha
        
        return best_h, best_score_local

    # Generate initial candidates
    candidates = []
    
    # 1. Random
    rng = np.random.default_rng(42)
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(8):
        phases = rng.uniform(0, 2*np.pi, size=8)
        amps = rng.uniform(0.3, 1.0, size=8)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 9):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    for block_size in [5, 10, 20, 40]:
        for i in range(3):
            h = np.zeros(N)
            for b in range(N // block_size):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            candidates.append(h)
    
    # 5. Walsh-Hadamard based
    # Generate random Walsh functions
    for i in range(5):
        # Random subset of Walsh functions
        n_funcs = rng.integers(3, 10)
        func_indices = rng.choice(N, size=n_funcs, replace=False)
        val = np.zeros(N)
        for idx in func_indices:
            # Walsh function: sign of product of (-1)^{bit_k(i)}
            bits = np.binary_repr(idx, width=int(np.log2(N)))
            w = np.ones(N)
            for bit_pos, bit in enumerate(bits):
                if bit == '1':
                    # Multiply by (-1)^{floor(i / 2^bit_pos) % 2}
                    w *= np.where((np.arange(N) >> bit_pos) % 2 == 0, 1, -1)
            val += w
        candidates.append(np.sign(val))

    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with local search and simulated annealing
    # Use local search for top 3, SA for next 3
    for i, (s, h) in enumerate(scored_candidates[:6]):
        if time.time() > deadline - 15:
            break
        
        if i < 3:
            # Local search
            time_limit = deadline - 10
            h_refined, s_refined = _local_search(h, time_limit)
        else:
            # Simulated annealing
            time_limit = deadline - 10
            h_refined, s_refined = _simulated_annealing(h, time_limit)
        
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do a few random restarts
    while time.time() < deadline - 5:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick local search with limited time
        time_limit = min(deadline - 2, time.time() + 3)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
