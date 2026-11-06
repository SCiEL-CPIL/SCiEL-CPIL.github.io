import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Initialize lpf array
    lpf = np.zeros(N_MAX + 2, dtype=np.int32)
    
    # Sieve
    for i in range(2, int(np.sqrt(N_MAX + 1)) + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i * i, N_MAX + 2, i):
                if lpf[j] == 0:
                    lpf[j] = i
            # Handle the case where i itself is the largest prime factor for multiples
            # The standard sieve above sets lpf[j] to the SMALLEST prime factor if we are not careful.
            # Wait, the original code sets lpf[j] = i for ALL multiples, overwriting with larger primes.
            # My NumPy draft above is for SMALLEST prime factor (if we break or use mask).
            # To get LARGEST prime factor, we must iterate primes in increasing order and overwrite.
            
            # Let's stick to the Python loop for correctness and simplicity, as N=10^6 is small.
            pass
            
    # The original Python loop is:
    # for i in range(2, N_MAX + 2):
    #     if lpf[i] == 0:
    #         for j in range(i, N_MAX + 2, i):
    #             lpf[j] = i
            
    # This is O(N log log N).
    # For N=10^6, this is fast.
    
    # Let's just optimize the original code slightly.
    lpf = [0] * (N_MAX + 2)
    for i in range(2, N_MAX + 2):
        if lpf[i] == 0:
            for j in range(i, N_MAX + 2, i):
                lpf[j] = i
                
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
