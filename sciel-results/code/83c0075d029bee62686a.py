"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    spf = list(range(MAX_N + 2))
    
    # Sieve of Eratosthenes - optimized
    limit = int(math.isqrt(MAX_N + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part(n) = product of p^e where e >= 2 in factorization
    pp = [1] * (MAX_N + 2)
    
    last_computed = MIN_N - 1
    for n in range(MIN_N, MAX_N + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        last_computed = n
        
        # Check time every 50000 iterations to reduce overhead
        if n % 50000 == 0 and time.time() > deadline:
            break
    
    # Find best n
    best_n = MIN_N
    best_val = 0
    
    # Only consider n where both pp[n] and pp[n+1] are computed
    end_range = min(MAX_N - 1, last_computed)
    
    for n in range(MIN_N, end_range + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
