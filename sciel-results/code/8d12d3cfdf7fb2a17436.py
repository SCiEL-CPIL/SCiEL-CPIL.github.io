import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80.0  # 80s deadline with 10s margin
    
    N = 1000000
    MAX_N = N + 1  # We need P[N+1]
    
    # Sieve of Eratosthenes for primes up to sqrt(MAX_N)
    limit = int(math.isqrt(MAX_N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Initialize powerful part array
    P = np.ones(MAX_N + 1, dtype=np.int64)
    
    # Fill P using vectorized operations
    for p in primes:
        q = p * p
        # q is p^k, starting with k=2
        while q <= MAX_N:
            # For all m = q * i such that i is not divisible by p
            max_i = MAX_N // q
            # Create array of valid i values (not divisible by p)
            i_vals = np.arange(1, max_i + 1)
            mask = (i_vals % p) != 0
            valid_i = i_vals[mask]
            indices = q * valid_i
            P[indices] *= q
            q *= p
    
    # Check time
    if time.time() > deadline - 5:
        # Fall back to simple scan
        P_list = P.tolist()
        best_n = 10000
        best_val = 0
        for n in range(10000, N + 1):
            val = P_list[n] * P_list[n + 1]
            if val > best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Find max product using vectorized operations
    P_list = P.tolist()
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = P_list[n] * P_list[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
