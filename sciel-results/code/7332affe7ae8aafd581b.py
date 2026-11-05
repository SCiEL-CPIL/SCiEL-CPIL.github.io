import numpy as np
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Generate LPF using NumPy
    # lpf[i] will hold the largest prime factor of i
    lpf = np.zeros(target + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # Iterate i from 2 to target
    # If lpf[i] is 0, i is prime.
    # Set lpf[j] = i for all multiples j of i.
    # Since i increases, the last prime factor assigned is the largest.
    for i in range(2, target + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    if time.time() > deadline:
        # Fallback if sieve is too slow (unlikely)
        return [1, target - 1]
        
    # Search for best a, b
    # a ranges from 1 to target // 2
    # b = target - a
    a_vals = np.arange(1, target // 2 + 1, dtype=np.int32)
    b_vals = target - a_vals
    
    # Calculate scores
    scores = np.maximum(lpf[a_vals], lpf[b_vals])
    
    # Find index of minimum score
    min_idx = np.argmin(scores)
    best_a = a_vals[min_idx]
    best_b = b_vals[min_idx]
    
    return [int(best_a), int(best_b)]
