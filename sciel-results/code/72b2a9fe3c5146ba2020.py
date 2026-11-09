import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s.
    # Sieve is ~1-2s. Scan is <0.1s. Total < 3s.
    deadline = start_time + 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # We need LPF up to 1000001
    
    # Precompute Largest Prime Factor for all numbers up to MAX_VAL
    # lpf[i] will hold the largest prime factor of i.
    # Initialize with 0.
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # Iterate i from 2 to MAX_VAL
    # If lpf[i] is 0, i is prime.
    # Mark all multiples of i with i.
    # Since i increases, the last mark is the largest prime factor.
    
    # Python loop for the outer sieve iteration is necessary because 
    # we need to check if lpf[i] is 0 to determine primality before 
    # slicing assignment. 
    # However, the inner loop can be vectorized.
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark multiples: i, 2i, 3i, ...
            # lpf[i::i] = i
            lpf[i::i] = i
            
    # Check time
    if time.time() > deadline:
        return N_MIN # Fallback, extremely unlikely
        
    # We need to find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    # n ranges from N_MIN to N_MAX
    # n+1 ranges from N_MIN+1 to N_MAX+1
    
    # Extract relevant slices
    # lpf_n corresponds to lpf[N_MIN : N_MAX+1]
    # lpf_np1 corresponds to lpf[N_MIN+1 : N_MAX+2]
    
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_np1 = lpf[N_MIN+1:N_MAX+2]
    
    # Calculate max for each pair
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    
    # The corresponding n is N_MIN + min_idx
    best_n = N_MIN + min_idx
    
    return int(best_n)
