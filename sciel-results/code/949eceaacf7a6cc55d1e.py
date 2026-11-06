import time
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # 5s margin
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    SIZE = N_PLUS_1 + 1
    
    # Initialize LPF array
    # Using int32 to save memory and potentially improve cache performance
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # We iterate i from 2 to N_PLUS_1
    # If lpf[i] is 0, i is prime.
    # We update all multiples of i with i.
    # Since we go in increasing order, the last write is the largest prime factor.
    
    # Optimization: Handle 2 separately?
    # lpf[2::2] = 2
    # Then loop i from 3 to N_PLUS_1 step 2?
    # No, we need to check if i is prime.
    # The standard loop is fine.
    
    for i in range(2, SIZE):
        if time.time() - start_time > TIME_LIMIT:
            # Fallback: return a known good candidate or the best found so far?
            # Since we are in the middle of the sieve, lpf is incomplete.
            # We cannot compute the score reliably.
            # However, the sieve is fast. This is unlikely to trigger.
            # If it does, we return a default safe value.
            return 10000
            
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    # Compute scores for n in [10000, 1000000]
    # n corresponds to lpf[n]
    # n+1 corresponds to lpf[n+1]
    
    # Slices
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Max LPF for the pair
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index with the minimum score
    min_idx = np.argmin(scores)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
