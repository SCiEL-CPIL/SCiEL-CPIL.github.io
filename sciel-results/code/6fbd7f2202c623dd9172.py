import numpy as np
import time

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
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Deterministic Construction: Van der Corput Sequence
    # This provides a low-discrepancy distribution of signs, aiming for spectral flatness.
    s = np.zeros(N)
    for k in range(1, N + 1):
        x = 0.0
        f = 0.5
        n = k - 1
        while n > 0:
            x += (n % 2) * f
            n //= 2
            f /= 2.0
        # Map [0, 1) to {-1, 1}
        if x < 0.5:
            s[k-1] = 1.0
        else:
            s[k-1] = -1.0
            
    s = s.astype(np.int8)
    
    # Initial evaluation
    A = _calc_acf(s)
    best_val = _max_abs_acf_from_array(A)
    best_seq = s.copy()
    
    # 2. Limited Local Refinement (Strictly Time-Boxed)
    # Try a small number of random flips to improve the deterministic baseline.
    current_s = best_seq.copy()
    current_val = best_val
    
    max_flips = 2000
    flips_done = 0
    
    while flips_done < max_flips:
        if time.time() > deadline - 5.0:
            break
            
        # Batch of 10 flips
        for _ in range(10):
            if flips_done >= max_flips:
                break
            if time.time() > deadline - 5.0:
                break
                
            idx = np.random.randint(0, N)
            old_val = current_s[idx]
            current_s[idx] = -old_val
            
            A_new = _calc_acf(current_s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val - 1e-12:
                current_val = val_new
            else:
                current_s[idx] = old_val
                
            flips_done += 1
        
        if current_val < best_val:
            best_val = current_val
            best_seq = current_s.copy()
            
    return best_seq
