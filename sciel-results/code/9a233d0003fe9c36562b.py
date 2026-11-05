import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 5s margin

    N_MAX = 1000001  # We need lpf up to n+1 where n can be up to 1000000
    
    # Use NumPy for efficient sieve
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(N_MAX + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # Iterate through potential primes
    for i in range(2, N_MAX + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark all multiples of i with i as their largest prime factor
            # Note: This overwrites previous smaller prime factors, which is correct
            # because we iterate i in increasing order, so the last prime that divides j is the largest.
            lpf[i::i] = i
            
        if time.time() > deadline:
            break

    # Define the range for n
    # n in [10000, 1000000]
    # We need lpf[n] and lpf[n+1]
    # n starts at 10000, ends at 1000000
    # n+1 starts at 100001, ends at 1000001
    
    # Create slices for n and n+1
    # lpf[10000 : 1000001] gives lpf[10000] ... lpf[1000000]
    # lpf[10001 : 1000002] gives lpf[10001] ... lpf[1000001]
    
    # Check if we have computed up to 1000001
    # The loop went to N_MAX (1000001), so yes.
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Calculate the objective value for each n
    # max(lpf[n], lpf[n+1])
    obj_vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the minimum value
    min_val = np.min(obj_vals)
    
    # Find the index of the minimum value
    # argmin returns the first occurrence
    min_idx = np.argmin(obj_vals)
    
    # Calculate n
    best_n = 10000 + min_idx
    
    return int(best_n)
