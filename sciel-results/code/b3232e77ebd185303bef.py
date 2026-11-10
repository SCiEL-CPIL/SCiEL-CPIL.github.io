import time
import sys
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    # We want to find M with many k such that k*sigma(k) = M
    # Strategy: Iterate k, compute M, count occurrences.
    # Limit k to N such that k*sigma(k) <= 10^18.
    # For k ~ 10^7, sigma(k) ~ k * log(log(k)) ~ 10^7 * 3 = 3*10^7.
    # M ~ 3*10^14, well below 10^18.
    # For k ~ 10^9, sigma(k) ~ 10^9 * 3 = 3*10^9.
    # M ~ 3*10^18, might exceed 10^18.
    # So k can go up to ~ 5*10^8 or more.
    # But iterating to 5*10^8 in Python is too slow for individual factorization or even sieve.
    # Let's try N = 10^7 first. If it's fast, we can try to increase.
    # However, we can't dynamically increase N in the middle of a single run easily without restructuring.
    # Let's pick N = 5 * 10^7.
    
    N = 50_000_000
    
    # Check if we have enough time
    if time.time() - start_time > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}

    # 1. Build Smallest Prime Factor (SPF) array
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma(k) for all k
    # sigma(k) can be computed from prime factorization.
    # If k = p1^a1 * p2^a2 * ..., sigma(k) = prod((p^(a+1)-1)/(p-1))
    # We can compute this iteratively or store it.
    # Storing sigma array of size N+1 (int64) is ~400MB for 5*10^7.
    # This might be too much memory.
    # Alternative: Compute M = k * sigma(k) on the fly and count in dict.
    # But we need to factorize each k.
    # Factorization using spf is fast.
    
    # Let's try to optimize memory.
    # We don't need to store sigma for all k, just process and count.
    # But we need to iterate k from 1 to N.
    
    counts = {}
    max_count = 0
    best_M = 0
    
    # Helper to compute sigma from spf
    def get_sigma(k):
        if k == 1:
            return 1
        n = k
        sigma_val = 1
        while n > 1:
            p = spf[n]
            power = 1
            sum_p = 1
            while n % p == 0:
                n //= p
                power *= p
                sum_p += power
            sigma_val *= sum_p
        return sigma_val

    # This function call overhead might be high for 5*10^7 iterations.
    # Let's inline the logic or optimize.
    
    # Actually, 5*10^7 iterations in Python is ~5-10 seconds if simple.
    # With factorization, it might be 20-50 seconds.
    # Let's try N = 10^7 first to be safe, then see.
    # If N=10^7 takes 5s, we can do N=5*10^7 in 25s.
    # Let's stick to N = 20_000_000 to be safe.
    
    N = 20_000_000
    
    # Rebuild SPF for smaller N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    counts = {}
    max_count = 0
    best_M = 0
    
    # Optimization: Local variables for speed
    spf_local = spf
    counts_local = counts
    limit_M = 10**18
    
    for k in range(1, N + 1):
        # Compute sigma(k)
        n = k
        sigma_val = 1
        while n > 1:
            p = spf_local[n]
            power = 1
            sum_p = 1
            while n % p == 0:
                n //= p
                power *= p
                sum_p += power
            sigma_val *= sum_p
        
        M = k * sigma_val
        if M <= limit_M:
            if M in counts_local:
                counts_local[M] += 1
                if counts_local[M] > max_count:
                    max_count = counts_local[M]
                    best_M = M
            else:
                counts_local[M] = 1
        
        # Time check
        if k % 1000000 == 0:
            if time.time() - start_time > deadline:
                break

    if best_M == 0:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}

    # Second pass to collect k's for best_M
    k_list = []
    for k in range(1, N + 1):
        n = k
        sigma_val = 1
        while n > 1:
            p = spf_local[n]
            power = 1
            sum_p = 1
            while n % p == 0:
                n //= p
                power *= p
                sum_p += power
            sigma_val *= sum_p
        
        M = k * sigma_val
        if M == best_M:
            k_list.append(k)
            
        if k % 1000000 == 0:
            if time.time() - start_time > deadline:
                break

    # Verify and format
    # k_list should be distinct by construction
    # Check k <= sqrt(M)
    # Check M <= 10^18
    
    return {'M': best_M, 'k': k_list}
