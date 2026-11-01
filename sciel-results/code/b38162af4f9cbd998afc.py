import numpy as np
import time

N = 400

def _compute_metric(h):
    """
    Compute the max absolute sidelobe of the autocorrelation.
    h is a 1D array of length N with values +/- 1.
    """
    # Linear autocorrelation via FFT
    # Zero-pad to 2N to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    # The main peak is at ac[0]. Sidelobes are ac[1:N].
    # Due to symmetry, ac[1:N] and ac[N+1:2N] are mirrors.
    # We check the first half.
    sidelobes = np.abs(ac[1:N])
    return np.max(sidelobes)

def _local_search(h, time_limit=5.0):
    """
    Local search via random bit flips.
    Returns the best sequence found.
    """
    start_time = time.time()
    current_h = h.copy()
    current_val = _compute_metric(current_h)
    best_h = current_h.copy()
    best_val = current_val
    
    # Precompute FFT size
    n_fft = 2 * N
    
    # Simple greedy local search with random restarts within the limit
    # We can try flipping bits that might reduce the max sidelobe.
    # A heuristic: identify the lag k where the max sidelobe occurs, and try to flip bits to reduce ac[k].
    # ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
    # Flipping h[j] changes ac[k] by -2*h[j]*(h[j+k] + h[j-k]) (with boundary checks).
    
    for _ in range(100000): # High iteration cap, controlled by time
        if time.time() - start_time > time_limit:
            break
            
        # Find the lag with the maximum sidelobe
        n_fft = 2 * N
        F = np.fft.rfft(current_h, n=n_fft)
        power = F * np.conj(F)
        ac = np.fft.irfft(power, n=n_fft)
        sidelobes = np.abs(ac[1:N])
        max_lag = np.argmax(sidelobes) + 1 # 1-indexed lag in ac array
        
        # Try flipping a bit that contributes significantly to ac[max_lag]
        # ac[max_lag] = sum_{i=0}^{N-1-max_lag} h[i] h[i+max_lag]
        # We want to change the sign of this sum.
        # Flipping h[j] affects terms where j is involved in the product.
        # If j < N - max_lag, term is h[j]h[j+max_lag]. Flip changes it to -h[j]h[j+max_lag].
        # Delta = -2 * h[j] * h[j+max_lag]
        # If j >= max_lag, term is h[j-max_lag]h[j]. Flip changes it to -h[j-max_lag]h[j].
        # Delta = -2 * h[j-max_lag] * h[j]
        
        # Let's just pick a random bit and see if it improves.
        # This is simple but effective with enough iterations.
        j = np.random.randint(N)
        old_h_j = current_h[j]
        current_h[j] = -old_h_j
        
        # Recompute metric
        F = np.fft.rfft(current_h, n=n_fft)
        power = F * np.conj(F)
        ac_new = np.fft.irfft(power, n=n_fft)
        new_val = np.max(np.abs(ac_new[1:N]))
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
        else:
            # Revert
            current_h[j] = old_h_j
            
    return best_h, best_val

def _generate_chirp(N):
    """
    Generate a chirp sequence.
    """
    i = np.arange(N)
    # Quadratic phase
    phase = np.pi * i**2 / N
    # Convert to binary +/- 1
    h = np.sign(np.cos(phase))
    # Handle zeros
    h[h == 0] = 1
    return h

def _generate_walsh(N):
    """
    Generate a Walsh-Hadamard sequence.
    N must be a power of 2 for standard Hadamard, but 400 is not.
    We can use a row of a larger Hadamard matrix or just random.
    Let's use a random binary sequence as a fallback for non-power-of-2.
    """
    return np.random.choice([-1, 1], size=N)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. Chirp
    candidates.append(_generate_chirp(N))
    
    # 3. Random sequences
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], size=N))
        
    # 4. Inverse Chirp
    i = np.arange(N)
    phase = -np.pi * i**2 / N
    h_inv = np.sign(np.cos(phase))
    h_inv[h_inv == 0] = 1
    candidates.append(h_inv)
    
    # Evaluate all candidates
    best_h = None
    best_val = float('inf')
    
    for h in candidates:
        val = _compute_metric(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Local search to polish the best candidate
    if time.time() < deadline - 10.0:
        # Use remaining time for local search
        time_limit = deadline - 10.0 - time.time()
        best_h, best_val = _local_search(best_h, time_limit=time_limit)
        
    return best_h
