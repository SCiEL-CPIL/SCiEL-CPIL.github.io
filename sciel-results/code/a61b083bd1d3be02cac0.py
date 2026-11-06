import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Use FFT to allow larger L
    L = 10**8
    
    # Generate powerful numbers
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    if time.time() > deadline - 50:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Prepare array for FFT
    # We need convolution of A with A.
    # A has length L+1.
    # Result length 2*L+1.
    # FFT size must be >= 2*L+1.
    n_fft = 1
    while n_fft < 2 * L + 1:
        n_fft *= 2
        
    # Create signal
    A = np.zeros(n_fft, dtype=np.float64)
    # Use numpy indexing to set 1s at powerful positions
    # powerfuls_list contains values <= L
    A[powerfuls_list] = 1.0
    
    # FFT
    F = np.fft.rfft(A)
    
    # Convolution in frequency domain
    C = F * F
    
    # Inverse FFT
    conv = np.fft.irfft(C, n=n_fft)
    
    # Round to integers
    # conv[i] represents count of pairs summing to i
    # We are interested in i from 2 to 2*L
    # conv is float, so we round
    conv_int = np.rint(conv[:2*L+1]).astype(np.int64)
    
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find max count
    # Ignore indices 0 and 1 (min sum is 1+1=2)
    # Actually, 1 is powerful, so 1+1=2 is valid.
    # But 0+0=0 is not valid (0 not powerful).
    # So we look at indices 2 to 2*L.
    if len(conv_int) < 3:
         return {'N': 2, 'pairs': [[1, 1]]}
         
    max_count = np.max(conv_int[2:])
    best_N = int(np.argmax(conv_int[2:])) + 2
    
    # Extract pairs
    powerfuls_set = set(powerfuls_list)
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
