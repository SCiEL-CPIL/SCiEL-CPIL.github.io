import time
import math
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    # The current approach is O(M log log M) for lpf computation, which is fast for M=1e6.
    DEADLINE = 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    M = N_MAX + 1  # We need lpf for values up to 1000001

    # Sieve of Eratosthenes using numpy
    # Using bytearray might be faster for the sieve part in pure Python, 
    # but we are using numpy for the lpf updates which are vectorized.
    is_prime = np.ones(M + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    limit = math.isqrt(M) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # Collect primes
    # np.where returns indices where condition is true
    primes = np.where(is_prime)[0]
    
    # Remove 0 and 1 if they were included (they are False, so they won't be in primes)
    # primes[0] is 2, primes[1] is 3, etc.
    
    # Compute largest prime factor for each number up to M
    # lpf[x] = largest prime factor of x
    lpf = np.zeros(M + 1, dtype=np.int32)
    
    # Iterate through primes. Since we process in increasing order, 
    # the last prime that divides a number is the largest prime factor.
    # lpf[multiple] = p overwrites previous values, so the largest p wins.
    for p in primes:
        # Generate multiples of p
        multiples = np.arange(p, M + 1, p)
        lpf[multiples] = p
        
        # Time check
        if time.time() - start_time > DEADLINE - 10.0:
            # If we are running out of time, we might not finish the lpf computation.
            # However, for M=1e6, this loop is fast. 
            # If we break here, lpf will be incomplete, leading to incorrect results.
            # It's better to let it finish if possible, or return a safe default if truly stuck.
            # Given the speed, this break is unlikely to trigger.
            pass

    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    # We can vectorize the search for the minimum as well to be extra safe on time
    # Create an array of max(lpf[n], lpf[n+1]) for n in range
    # n ranges from N_MIN to N_MAX
    # lpf[n] and lpf[n+1]
    # Let's create a slice
    # lpf[N_MIN : N_MAX + 1] and lpf[N_MIN + 1 : N_MAX + 2]
    
    lpf_n = lpf[N_MIN : N_MAX + 1]
    lpf_np1 = lpf[N_MIN + 1 : N_MAX + 2]
    
    # Compute max for each pair
    # np.maximum works element-wise
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = N_MIN + min_idx
    best_val = max_lpf[min_idx]
    
    return int(best_n)
