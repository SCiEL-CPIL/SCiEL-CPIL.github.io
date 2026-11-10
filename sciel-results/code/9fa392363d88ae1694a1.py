import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need pp up to N+1
    
    # Array to store powerful parts, 1-indexed
    pp = [1] * (max_val + 1)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, compute its contribution to powerful parts
    # Optimized: use incremental power upgrades
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        # For all multiples of p^2: multiply by p^2
        for m in range(p2, max_val + 1, p2):
            pp[m] *= p2
        
        # For all multiples of p^3: multiply by p (upgrade to p^3)
        pk = p2 * p
        while pk <= max_val:
            for m in range(pk, max_val + 1, pk):
                pp[m] *= p
            pk *= p
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Now find the max product pp[n] * pp[n+1] for n in [10000, 1000000]
    best_n = 10000
    max_prod = 0
    
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
        
        # Check deadline periodically
        if (n - 10000) % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n
