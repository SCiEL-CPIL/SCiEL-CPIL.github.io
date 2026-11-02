import numpy as np
import time

N = 400

def _autoconv_fft(s):
    # Use FFT for fast auto-correlation
    # Zero pad to next power of 2 or 2*N for linear convolution
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    R = np.fft.irfft(S * S.conj(), n=n_fft)
    # The first N elements are the auto-correlation for lags 0 to N-1
    # We need lags 0 to N-1. 
    # Note: irfft returns real part.
    return R[:N]

def _score(s):
    # Calculate max absolute auto-correlation excluding lag 0
    R = _autoconv_fft(s)
    # R[0] is N. We ignore it for the "sidelobe" metric if that's the goal.
    # If the metric includes lag 0, it's always N.
    # Assuming the task wants to minimize sidelobes.
    if len(R) > 1:
        return np.max(np.abs(R[1:]))
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a random balanced sequence
    # To ensure balance, start with 200 ones and 200 minuses
    s = np.ones(N)
    s[N//2:] = -1
    np.random.shuffle(s)
    
    current_score = _score(s)
    best_s = s.copy()
    best_score = current_score
    
    # Hill Climbing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip
        s[i] *= -1
        
        new_score = _score(s)
        
        # Simulated Annealing acceptance
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_s = s.copy()
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                s[i] *= -1
        
        T *= cooling
        
    return best_s
