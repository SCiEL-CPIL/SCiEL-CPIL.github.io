"""Solve: maximize powerful_part(n)*powerful_part(n+1) for n in [10000, 1000000].

The powerful part of m is the product of p^e over primes p with exponent e>=2 in m's factorization.
We precompute powerful_part for all m in [1, 1000001] via sieve, then scan for the max product.
"""
import time

def construct():
    t_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 1000000
    # We need powerful_part for n and n+1 where n in [10000, 1000000]
    # So we need powerful_part for values in [10000, 1000001]
    MAXV = N + 1  # 1000001

    # powerful_part[i] will hold the powerful part of i
    # Initialize to 1 for all
    pp = [1] * (MAXV + 1)

    # Sieve: for each prime p, for each power p^e with e>=2, multiply pp[i] by p^e
    # for all i that are multiples of p^e but not p^(e+1)? 
    # Actually, the powerful part of n is the product over primes p|n of p^e where e is the
    # exact exponent of p in n, IF e>=2. So we need to compute the exact exponent.
    
    # Better approach: for each prime p, for each multiple i of p, compute the exponent of p in i.
    # Then if exponent >= 2, multiply pp[i] by p^exponent.
    
    # Even better: use a sieve to compute the exponent of each prime in each number.
    # Let exp[i] be a dict or we can do it per-prime.
    
    # Most efficient: for each prime p, iterate through multiples. For each multiple i of p,
    # compute v_p(i) by repeatedly dividing. But that's slow for large numbers.
    
    # Alternative: for each prime p, for each power q = p^e (e>=2), for each multiple i of q,
    # we know p^e divides i. But we need the EXACT power. 
    # 
    # Let me think differently. For each i, powerful_part(i) = product of p^e where e = v_p(i) >= 2.
    # 
    # Efficient sieve approach:
    # 1. Compute smallest prime factor (spf) for all numbers up to MAXV.
    # 2. For each number i, factorize using spf and compute powerful part.
    #    But factoring each number individually is O(MAXV * log(MAXV)) which might be too slow.
    
    # Let's use the spf approach but vectorize with NumPy where possible.
    
    import numpy as np
    
    # Compute smallest prime factor for all numbers up to MAXV
    spf = np.arange(MAXV + 1, dtype=np.int64)
    for i in range(2, int(np.sqrt(MAXV)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAXV + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute powerful_part for each number using spf
    # For each i from 10000 to MAXV:
    #   Factorize i using spf, compute product of p^e for e>=2
    
    # This is O(MAXV * log(MAXV)) in the worst case, but with spf it's fast.
    # Let's do it in a loop but check time periodically.
    
    pp = [1] * (MAXV + 1)
    
    for i in range(10000, MAXV + 1):
        if time.time() - t_start > DEADLINE:
            break
        x = i
        prod = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                prod *= p ** e
        pp[i] = prod
    
    # Find max of pp[n] * pp[n+1] for n in [10000, 1000000]
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
