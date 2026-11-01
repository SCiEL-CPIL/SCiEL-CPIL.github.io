import numpy as np
import time

N = 400
M = 2 * N  # FFT size for full autocorrelation

def _calc_score_fft(h):
    """
    Calculate the score using FFT-based autocorrelation.
    Score = max_{k != 0} |C[k]| / ||h||^2
    """
    # Pad to size M for circular convolution = linear autocorrelation
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    # FFT
    H = np.fft.fft(h_padded)
    # Autocorrelation via power spectrum
    C = np.fft.ifft(H * np.conj(H)).real
    
    # C[0] is lag 0 (peak), C[1..N-1] are positive lags, C[M-N+1..M-1] are negative lags
    # For real sequences, C is symmetric: C[k] = C[M-k]
    # We want max |C[k]| for k != 0
    # C[0] is lag 0
    # C[1..N-1] are lags 1..N-1
    # C[M-N+1..M-1] = C[N+1..M-1] are lags -(N-1)..-1 (by symmetry)
    
    # Extract all lags except 0
    # Positive lags: C[1:N]
    # Negative lags: C[M-N+1:M] (which are C[N+1..M-1], corresponding to lags -(N-1)..-1)
    
    pos_lags = C[1:N]  # lags 1 to N-1
    neg_lags = C[M-N+1:M]  # lags -(N-1) to -1
    
    all_lags = np.concatenate([pos_lags, neg_lags])
    peak = np.max(np.abs(all_lags))
    
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _calc_score_direct(h):
    """
    Fallback: direct correlation calculation.
    """
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _initialization_strategy(seed):
    """
    Generate diverse initial sequences using different structured patterns.
    """
    rng = np.random.RandomState(seed)
    
    strategy = seed % 6
    
    if strategy == 0:
        # Random
        return rng.choice([-1, 1], size=N)
    
    elif strategy == 1:
        # Alternating with random flips
        h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        flip_mask = rng.rand(N) < 0.1
        h[flip_mask] *= -1
        return h
    
    elif strategy == 2:
        # Block-based: random blocks of +/- 1
        block_sizes = rng.randint(5, 25, size=20)
        h = np.zeros(N, dtype=int)
        pos = 0
        sign = 1 if rng.rand() > 0.5 else -1
        for bs in block_sizes:
            end = min(pos + bs, N)
            h[pos:end] = sign
            pos = end
            sign *= -1
        if pos < N:
            h[pos:] = sign
        return h
    
    elif strategy == 3:
        # Low-discrepancy-like: use golden ratio sequence
        golden = (1 + np.sqrt(5)) / 2
        t = np.arange(N)
        x = (t * (1.0 / golden)) % 1.0
        h = np.where(x > 0.5, 1, -1).astype(int)
        # Add some randomness
        flip_mask = rng.rand(N) < 0.05
        h[flip_mask] *= -1
        return h
    
    elif strategy == 4:
        # Walsh-Hadamard-like: use a random Hadamard row
        # Simple version: use bits of a random number
        h = np.zeros(N, dtype=int)
        for i in range(N):
            # Use a pseudo-random bit sequence
            val = rng.randint(0, 2)
            h[i] = 1 if val == 1 else -1
        return h
    
    else:
        # Logistic map based
        r = 3.57 + 0.43 * rng.rand()
        x = 0.1 + 0.8 * rng.rand()
        h = np.zeros(N, dtype=int)
        for j in range(N):
            x = r * x * (1 - x)
            h[j] = 1 if x > 0.5 else -1
        return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Try multiple diverse initializations
    num_initial = 30
    for i in range(num_initial):
        if time.time() > deadline - 30:
            break
        
        h = _initialization_strategy(i * 7 + 1)
        score = _calc_score_fft(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Phase 2: Simulated annealing from the best initial
    if best_h is not None:
        h = best_h.copy()
        current_score = best_score
        
        # Annealing parameters
        T_start = 0.05
        T_end = 0.001
        cooling_rate = 0.995
        
        T = T_start
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 10 and T > T_end and no_improve < max_no_improve:
            # Pick a random position to flip
            j = np.random.randint(N)
            old_val = h[j]
            h[j] = -old_val
            
            new_score = _calc_score_fft(h)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                h[j] = old_val
                no_improve += 1
            
            # Cool down
            T *= cooling_rate
        
        # Phase 3: Final hill climbing from the best state
        h = best_h.copy()
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                
                old_val = h[j]
                h[j] = -old_val
                new_score = _calc_score_fft(h)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h.copy()
                else:
                    h[j] = old_val
    
    # Fallback
    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    
    return best_h
