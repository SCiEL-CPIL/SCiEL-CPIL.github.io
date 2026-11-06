import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # Seconds, leave 5s margin
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    
    # 1. Compute Largest Prime Factor for all numbers up to 1000001
    # lpf[x] will store the largest prime factor of x
    lpf = [0] * (N_PLUS_1 + 1)
    
    # Sieve of Eratosthenes variant for LPF
    # Iterate through potential primes
    for i in range(2, N_PLUS_1 + 1):
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i
            for j in range(i, N_PLUS_1 + 1, i):
                lpf[j] = i
                
    # Convert to numpy array for vectorized search
    lpf_np = np.array(lpf, dtype=np.int32)
    
    # 2. Search for optimal n
    # We need n in [10000, 1000000]
    # score(n) = max(lpf[n], lpf[n+1])
    
    # Slices for n and n+1
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 10001 to 1000001
    
    lpf_n = lpf_np[10000:N_MAX + 1]
    lpf_np1 = lpf_np[10001:N_MAX + 2]
    
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    best_n = 10000 + min_idx
    
    return best_n
# EVOLVE-BLOCK-END
