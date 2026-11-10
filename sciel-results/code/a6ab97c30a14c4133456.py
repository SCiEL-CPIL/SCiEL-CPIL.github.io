import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for evaluator overhead
    
    max_val = 1000001
    
    # Initialize LPF array
    # We use a 32-bit integer array to save memory and potentially speed up operations
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve to find largest prime factors
    # For each prime p, it is the largest prime factor for all multiples of p
    # that are not divisible by any prime > p.
    # Since we iterate p in increasing order, if we simply assign lpf[j] = p 
    # for all multiples j, the last assignment (largest p) will remain.
    
    # We need to identify primes. A simple sieve of Eratosthenes can be used,
    # or we can just iterate and check if lpf[p] == 0 (meaning p hasn't been marked by a smaller prime).
    # However, the standard approach for LPF sieve is:
    # for p in range(2, max_val + 1):
    #     if lpf[p] == 0: # p is prime
    #         for j in range(p, max_val + 1, p):
    #             lpf[j] = p
    
    # Vectorized version:
    # We iterate through potential primes.
    # To check if p is prime, we can check if lpf[p] == 0.
    # But wait, if we just iterate p from 2 to max_val, and lpf[p] is 0, then p is prime.
    # Then we update multiples.
    
    # This loop is still Python-level for the primes, but the inner update is vectorized.
    # Number of primes up to 1,000,000 is ~78,000.
    # The total number of operations in the inner loop is sum(max_val/p) ~ max_val * log(log(max_val)).
    # This is efficient.
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            # Update multiples of p
            # indices: p, 2p, 3p, ...
            # We can create an array of indices or use slicing
            # Slicing: lpf[p::p] = p
            # This is a vectorized assignment
            lpf[p::p] = p
            
        # Time check every 10000 iterations
        if p % 10000 == 0 and time.time() > deadline:
            return 10000

    if time.time() > deadline:
        return 10000

    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # We can do this with vectorized operations
    
    # Create arrays for n and n+1
    # n ranges from 10000 to 1000000
    # lpf_n = lpf[10000:1000001]
    # lpf_np1 = lpf[10001:1000002]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Compute max for each pair
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    min_score = scores[min_idx]
    
    # The corresponding n is 10000 + min_idx
    best_n = 10000 + min_idx
    
    # Optional: Early termination check in the original code was for min_score <= 5.
    # With vectorization, we can just return the result.
    # The minimum possible LPF for numbers > 1 is 2 (for powers of 2).
    # For consecutive integers, one is even. The other could be 3, 5, etc.
    # The minimum max(LPF(n), LPF(n+1)) is likely small.
    # For example, if n is a power of 2, LPF(n)=2. If n+1 is 3, LPF=3. Max=3.
    # If n+1 is 5, Max=5.
    # We just return the best found.
    
    return int(best_n)
