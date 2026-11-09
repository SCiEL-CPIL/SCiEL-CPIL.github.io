"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 1000001  # we need up to 1,000,001 to compute powerful_part for n and n+1 where n in [10000, 1000000]
    
    # Sieve-like approach to compute the powerful part for each number
    # powerful_part(n) = product of p^e for all prime powers p^e || n with e >= 2
    
    # We'll use a modified sieve to track the highest power of each prime dividing each number
    # For each number, we want the product of p^e where e is the exact exponent of p in n, IF e >= 2
    
    # Approach:
    # 1. For each prime p, for each power p^k (k>=2), mark all multiples of p^k
    # 2. But we need the FULL p^e, not just p^2. So we need to know the exact exponent.
    
    # Better approach: Use smallest prime factor (SPF) sieve, then factor each number using SPF
    # But that might be slow for 1M numbers. Let's think of a sieve-based way.
    
    # Alternative: 
    # powerful_part[n] = 1 initially
    # For each prime p:
    #   p2 = p*p
    #   For each multiple m of p2:
    #     Determine the exact exponent e of p in m
    #     powerful_part[m] *= p^e
    # 
    # To determine exact exponent efficiently, we can precompute for each number its
    # "prime power contribution" using a sieve.
    
    # Let's use: spf (smallest prime factor) array via linear sieve, then factor each number.
    # Actually, let's just do a direct sieve for powerful parts.
    
    import numpy as np
    
    N = 1000001
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # For each prime p, consider p^2, p^3, etc.
    # For each number m, we want to multiply by p^e where e is the exact exponent of p in m, if e >= 2.
    
    # Sieve for primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            is_prime[i::i] = False
    
    # For each prime p, find all powers p^k for k >= 2, and for each such power,
    # iterate over multiples and determine the exact exponent.
    # This is tricky. Let's use a different approach:
    
    # For each number m from 2 to N, we can compute its powerful part by factoring it.
    # But 1M factorizations might be too slow in pure Python. Let's use numpy where possible.
    
    # Alternative efficient sieve:
    # powerful[m] = 1
    # For each prime p:
    #   p2 = p * p
    #   For each multiple m of p2:
    #     Count how many times p divides m (i.e., the exponent e)
    #     If e >= 2, powerful[m] *= p**e
    
    # To count exponent of p in m: while m % p == 0: e++; m //= p
    # But doing this for every multiple of p^2 for every prime is O(N log log N * something)
    
    # Let's just do it with a loop. N = 1,000,001. Number of primes ~ 78,498.
    # For each prime p, the number of multiples of p^2 up to N is N/p^2.
    # Sum over primes of N/p^2 ≈ N * sum(1/p^2) ≈ N * 0.4522 ≈ 452,200.
    # For each such multiple, we need to count the exponent, which takes at most log_p(m) steps.
    # This should be fast enough.
    
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each multiple of p2
        for m in range(p2, N + 1, p2):
            # Count exponent of p in m
            temp = m
            e = 0
            while temp % p == 0:
                e += 1
                temp //= p
            if e >= 2:
                powerful[m] *= (p ** e)
    
    # Now find n in [10000, 1000000] that maximizes powerful[n] * powerful[n+1]
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]
    
    for n in range(10000, 1000000):
        val = powerful[n] * powerful[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return int(best_n)
# EVOLVE-BLOCK-END
