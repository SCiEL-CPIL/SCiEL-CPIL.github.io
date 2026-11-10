import time
import numpy as np

def construct():
    start_time = time.time()
    limit = 1000000
    max_val = limit + 1
    
    # Step 1: Generate primes up to max_val using Sieve of Eratosthenes
    # is_prime[i] will be True if i is prime
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # Step 2: Compute Largest Prime Factor (LPF) for all numbers up to max_val
    # lpf[k] will store the largest prime factor of k
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Iterate through primes in increasing order.
    # For each prime p, assign p to all multiples of p.
    # Since we go in increasing order, the last assignment for a number k
    # will be its largest prime factor.
    for p in primes:
        lpf[p::p] = p
        
    # Check time
    if time.time() - start_time > 45:
        # Fallback, though this step is very fast
        return 10000

    # Step 3: Find n in [10000, 1000000] minimizing max(LPF(n), LPF(n+1))
    # n ranges from 10000 to 1000000
    # n+1 ranges from 10001 to 1000001
    
    n_vals = lpf[10000:1000001]
    np1_vals = lpf[10001:1000002]
    
    scores = np.maximum(n_vals, np1_vals)
    
    min_idx = np.argmin(scores)
    best_n = 10000 + min_idx
    
    return int(best_n)
