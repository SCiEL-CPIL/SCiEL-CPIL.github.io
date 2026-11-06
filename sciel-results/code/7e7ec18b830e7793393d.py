import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    max_n = 1000000
    N = max_n
    
    # pp[i] will store the powerful part of i
    pp = [1] * (N + 1)
    
    # Sieve for primes up to sqrt(N) is enough for the outer loop?
    # No, p can be up to N, but p^2 <= N implies p <= sqrt(N).
    # So we only need primes up to sqrt(N) = 1000.
    limit = int(math.isqrt(N))
    
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
        
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
                
    # Accumulate powerful parts
    for p in primes:
        val = p * p
        while val <= N:
            next_val = val * p
            
            # Iterate through multiples of val
            # m = val, 2*val, ...
            # Check if m is divisible by next_val.
            # m = val * k. m % next_val == 0 iff (val * k) % (val * p) == 0 iff k % p == 0.
            
            k = 1
            for m in range(val, N + 1, val):
                if k % p != 0:
                    pp[m] *= val
                k += 1
                
            val = next_val

    # Find the best n
    best_n = 10000
    best_val = -1
    
    # n must be in [10000, 999999] so that n+1 <= 1000000
    for n in range(10000, N):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
