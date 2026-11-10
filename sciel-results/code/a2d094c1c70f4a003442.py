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
    
    # Precompute powerful parts for all numbers up to MAX_N
    # Using a sieve-like approach for efficiency
    pp = [1] * (MAX_N + 2)
    
    # For each prime p, for each power p^e where e >= 2 and p^e <= MAX_N,
    # mark all multiples of p^e as having p^e in their powerful part
    # But we need to be careful: the powerful part is the product of ALL prime powers with e>=2
    # So for each number n, pp[n] = product of p^e for all p^e || n where e >= 2
    
    # Alternative: factor each number using SPF and compute pp
    # Let's use the SPF approach but optimize it
    
    # Precompute factorization info for all numbers
    # pp[n] will be the powerful part of n
    for i in range(2, MAX_N + 1):
        if spf[i] == i:  # i is prime
            p = i
            p2 = p * p
            while p2 <= MAX_N:
                # All multiples of p2 have p^2 in their factorization
                # We need to determine the full exponent of p in each multiple
                # This is complex; let's use direct factorization instead
                break
    
    # Direct factorization approach with SPF - optimized
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
    
    # Compute powerful parts for range [MIN_N, MAX_N+1]
    # We need pp[n] for n in [MIN_N, MAX_N] and pp[n+1] for n in [MIN_N, MAX_N-1]
    # So compute pp for [MIN_N, MAX_N]
    
    pp_arr = [1] * (MAX_N + 2)
    
    for i in range(MIN_N, MAX_N + 1):
        pp_arr[i] = powerful_part(i)
        
        # Check deadline periodically
        if (i - MIN_N) % 10000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Find the best consecutive pair
    best_n = MIN_N
    best_product = 0
    
    for i in range(MIN_N, MAX_N):
        if time.time() > deadline - 2:
            break
        prod = pp_arr[i] * pp_arr[i + 1]
        if prod > best_product:
            best_product = prod
            best_n = i
    
    return best_n
# EVOLVE-BLOCK-END
