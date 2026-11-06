import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # We scan a large contiguous range to find the longest run of distinct totients.
    # 50,000,000 is a good balance between computation time and coverage.
    N = 50_000_000
    
    # 1. Compute phi(1..N) using NumPy segmented sieve logic (actually full sieve since N is manageable)
    phi = np.arange(1, N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N)
    limit = int(math.isqrt(N))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # Update phi for each prime
    for p in primes:
        # Index i in phi corresponds to number i+1.
        # We want to update numbers divisible by p.
        # First number divisible by p is p, which is at index p-1.
        # Step is p.
        start_idx = p - 1
        # phi[j] -= phi[j] // p
        # This operation is vectorized.
        phi[start_idx::p] -= phi[start_idx::p] // p
        
    # 2. Find longest run of distinct values using sliding window
    last_seen = {}
    left = 0
    max_k = 0
    best_s = 1
    
    # Local variables for speed
    ls = last_seen
    ph = phi
    
    for i in range(N):
        val = int(ph[i])
        if val in ls:
            prev = ls[val]
            if prev >= left:
                left = prev + 1
        ls[val] = i
        k = i - left + 1
        if k > max_k:
            max_k = k
            best_s = left + 1 # 1-based start index
            
            # Early exit if we reach the maximum possible length
            if max_k == 200:
                break
                
    # If we have time, we could potentially search higher ranges, 
    # but the dense scan is the most robust strategy for this constraint.
    
    return {'start': best_s, 'length': max_k}
