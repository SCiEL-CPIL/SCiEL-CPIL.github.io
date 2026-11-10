"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to compute powerful part using SPF
    def powerful_part(n):
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        return pp
    
    best_n = MIN_N
    best_product = 0
    
    # Precompute powerful parts for range [MIN_N, MAX_N]
    # We need pp[n] for n in [MIN_N, MAX_N] and pp[n+1] for n in [MIN_N, MAX_N-1]
    # So we compute pp for [MIN_N, MAX_N]
    pp = [1] * (MAX_N + 2)
    
    for i in range(MIN_N, MAX_N + 1):
        pp[i] = powerful_part(i)
        
        # Check deadline periodically
        if (i - MIN_N) % 50000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Find the best consecutive pair
    for i in range(MIN_N, MAX_N):
        if time.time() > deadline - 2:
            break
        prod = pp[i] * pp[i + 1]
        if prod > best_product:
            best_product = prod
            best_n = i
    
    return best_n
# EVOLVE-BLOCK-END
