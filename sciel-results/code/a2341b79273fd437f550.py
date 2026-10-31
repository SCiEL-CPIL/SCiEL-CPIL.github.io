"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(s):
    """Compute the full autoconvolution of a 1D array using FFT."""
    n = len(s)
    size = 2 * n - 1
    f = np.fft.rfft(s, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _objective(signed_seq):
    """
    Compute the maximum absolute value of the off-peak autoconvolution.
    This aligns with the 'C3-prime' task description focusing on off-peak bounds
    and the 'raw value' metric which likely excludes the trivial DC peak.
    """
    ac = _autoconv(signed_seq)
    # Exclude the peak at index 0 (lag 0)
    if len(ac) > 1:
        return np.max(np.abs(ac[1:]))
    return 0.0

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

def construct():
    # 1. Generate base sequence from Legendre symbols of a prime p > N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p) or p <= N:
        p += 1

    base_seq = np.array([_legendre_symbol(i + 1, p) for i in range(N)], dtype=float)

    # 2. Local Search to minimize off-peak max absolute autocorrelation
    current_seq = base_seq.copy()
    current_obj = _objective(current_seq)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    max_iterations = 5000
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        best_gain = 0
        best_index = -1
        
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            # Flip sign at i
            current_seq[i] *= -1
            new_obj = _objective(current_seq)
            gain = current_obj - new_obj
            
            if gain > best_gain:
                best_gain = gain
                best_index = i
            else:
                # Revert
                current_seq[i] *= -1
                continue
            
            # Revert to keep evaluating single flips from current state
            current_seq[i] *= -1
        
        if best_index != -1:
            current_seq[best_index] *= -1
            current_obj -= best_gain
            improved = True
        
        iter_count += 1

    return current_seq
# EVOLVE-BLOCK-END
