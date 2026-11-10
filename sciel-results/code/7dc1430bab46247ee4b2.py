"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 2, i):
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
    
    # Precompute powerful parts for range to avoid recomputing
    # We need pp[n] for n in [MIN_N, MAX_N]
    pp_arr = [1] * (MAX_N + 2)
    for i in range(MIN_N, MAX_N + 1):
        pp_arr[i] = powerful_part(i)
    
    for n in range(MIN_N, MAX_N):
        prod = pp_arr[n] * pp_arr[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
