import numpy as np
import time

def construct():
    # Start time tracking for safety margin
    start_time = time.time()
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    SIZE = N_PLUS_1 + 1  # 1,000,002
    
    # Initialize LPF array with int32 (max LPF < 1,000,000)
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Handle factor 2
    lpf[2::2] = 2
    
    # Sieve for odd numbers
    # We iterate i from 3 to SIZE step 2
    # If lpf[i] is 0, i is prime
    # Mark multiples of i with i (overwriting smaller prime factors)
    # This results in lpf[x] being the largest prime factor of x
    
    # Optimization: Only need to sieve up to SIZE
    # The loop naturally handles all primes
    for i in range(3, SIZE, 2):
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    # Calculate scores for n in [10000, 1000000]
    # lpf[n] is LPF of n
    # lpf[n+1] is LPF of n+1
    # Score is max(lpf[n], lpf[n+1])
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    scores = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(scores)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
