import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.random.randint(0, 2, size=N) * 2 - 1
    
    # Simulated Annealing with Block Flips
    # Initial temperature
    T = 5.0
    # Cooling rate
    alpha = 0.99995
    
    current_seq = best_seq.copy()
    current_acf = _calc_acf(current_seq)
    current_val = _max_abs_acf_from_array(current_acf)
    
    # To speed up, we don't recompute the full FFT for every tiny change if we could, 
    # but with N=400, FFT is ~microseconds. 
    # A single FFT of 1024 points is very fast.
    # Let's estimate: 100,000 iterations * 10us = 1 second. 
    # We have 115 seconds, so we can do ~10 million iterations.
    
    # However, Python overhead for loop + numpy call is significant.
    # Let's optimize the inner loop.
    
    # We will use a mix of small and large block flips.
    
    while time.time() < deadline:
        # Generate a block flip
        # Sample length L
        # Favor small L but allow larger
        # L = 1 with prob 0.5, L in [2, 10] with prob 0.4, L in [11, 50] with prob 0.1
        r = np.random.random()
        if r < 0.5:
            L = 1
        elif r < 0.9:
            L = np.random.randint(2, 11)
        else:
            L = np.random.randint(11, 51)
        
        # Sample start index
        # Ensure block stays within bounds
        max_start = N - L
        if max_start < 0:
            continue
        i = np.random.randint(0, max_start + 1)
        
        # Flip the block
        current_seq[i:i+L] *= -1
        
        # Calculate new autocorrelation
        new_acf = _calc_acf(current_seq)
        new_val = _max_abs_acf_from_array(new_acf)
        
        # Metropolis Acceptance
        delta_E = new_val - current_val
        
        if delta_E < 0:
            # Accept
            current_val = new_val
            current_acf = new_acf
            
            if current_val < best_val:
                best_val = current_val
                best_seq = current_seq.copy()
        else:
            # Reject or accept with prob exp(-delta_E / T)
            if T > 0:
                prob = math.exp(-delta_E / T)
                if np.random.random() < prob:
                    # Accept
                    current_val = new_val
                    current_acf = new_acf
                    if current_val < best_val:
                        best_val = current_val
                        best_seq = current_seq.copy()
                else:
                    # Reject: Undo the flip
                    current_seq[i:i+L] *= -1
            else:
                # Reject: Undo the flip
                current_seq[i:i+L] *= -1
        
        # Cool down
        T *= alpha
        
        # If T is very low, we might want to restart with a slightly higher T 
        # or just continue. Given the time budget, continuing is fine.
        # But if T < 0.001, we are effectively doing greedy.
        # Let's keep T above a small floor to allow some noise if stuck?
        # Or just let it go to 0.
        
    return best_seq
