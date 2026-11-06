"""
Strategy: Exhaustive Precomputation via Sieve.
Instead of searching for specific candidates using CRT, we precompute the powerful part
for every integer in the valid range [10000, 1000000].
We use a modified sieve to accumulate prime powers p^e (e>=2) into an array.
Then we simply scan the array for the maximum product of consecutive powerful parts.
This guarantees finding the global optimum within the range, provided the sieve is efficient.
"""

import time
import math
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Deadline guard: 90s limit, keep 5s margin
    deadline = start_time + 85.0
    
    # Range for n: [10000, 999999] so n+1 <= 1000000
    # We need powerful parts for numbers up to 1000000.
    N_MAX_VAL = 1000000
    N_MIN = 10000
    N_MAX = N_MAX_VAL - 1 # n can be up to 999999
    
    # Initialize powerful part array with 1s
    # powerful_part[i] will store the product of all prime powers p^e (e>=2) dividing i
    pp = np.ones(N_MAX_VAL + 1, dtype=np.int64)
    
    # We need to iterate over primes.
    # A prime p contributes p^e to pp[i] if p^e divides i.
    # We can do this by iterating p, then p^2, p^3, ... and adding to multiples.
    # Actually, it's simpler to just compute the factorization for each number? No, that's O(N log N) or worse.
    # Sieve approach:
    # For each prime p:
    #   Let p2 = p*p
    #   For each multiple m of p2:
    #     We need to determine the full exponent e of p in m.
    #     This is tricky to do incrementally in a simple sieve without division.
    
    # Alternative: Standard SPF sieve, then one pass to compute pp.
    # SPF sieve is O(N log log N).
    # Computing pp from SPF is O(N log N) in worst case, but average is small.
    # Given N=1e6, this is very fast in Python if optimized, or NumPy if possible.
    # However, computing pp from SPF in a loop in Python might be slow (1e6 iterations * factorization).
    # Let's try the direct sieve accumulation method, which is O(N log log N) effectively.
    
    # Method:
    # 1. Find all primes up to N_MAX_VAL using a simple sieve of Eratosthenes.
    # 2. For each prime p:
    #    Compute p^2, p^3, ... up to N_MAX_VAL.
    #    For each power pk = p^k (k>=2):
    #       For each multiple m of pk:
    #          We want to multiply pp[m] by p^(e - (k-1))? No.
    #          This approach is complicated because we need the *full* power.
    
    # Let's stick to SPF + Vectorized/Loop computation.
    # With N=1e6, a pure Python loop for factorization might take ~1-2 seconds.
    # 1e6 numbers, avg 3-4 prime factors. ~4e6 operations. Should be well under 85s.
    
    # Step 1: Build SPF
    spf = np.zeros(N_MAX_VAL + 1, dtype=np.int64)
    for i in range(2, N_MAX_VAL + 1):
        if spf[i] == 0:
            # i is prime
            spf[i] = i
            if i * i <= N_MAX_VAL:
                for j in range(i * i, N_MAX_VAL + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
                    
    # Ensure spf for composites that weren't set (should be all set by above)
    # Actually, the above sets spf for composites >= i*i.
    # Primes < sqrt(N) set their multiples.
    # What about composites with a prime factor > sqrt(N)?
    # If n is composite, it has a prime factor <= sqrt(n) <= sqrt(N).
    # So all composites are covered.
    # But spf[prime] is set to prime.
    
    # Step 2: Compute powerful part for each i
    # pp[i] = product of p^e where e >= 2 in factorization of i.
    
    # We can do this efficiently by iterating i from 2 to N_MAX_VAL.
    # However, doing this in a pure Python loop might be borderline.
    # Let's try to optimize.
    
    # Actually, we can compute pp[i] using the recurrence:
    # Let i = p^k * m where p does not divide m.
    # If k >= 2, pp[i] = p^k * pp[m].
    # If k == 1, pp[i] = pp[m].
    # This requires knowing the smallest prime factor and the rest.
    
    # Let's use the SPF array to compute pp in a single pass.
    # We process i from 2 to N_MAX_VAL.
    # spf[i] is the smallest prime factor.
    # Let p = spf[i].
    # Let m = i // p.
    # We need to know the exponent of p in i.
    # This is hard to do in O(1) without extra arrays.
    
    # Alternative: Just factor each number using spf.
    # It's O(log i) per number. Total O(N log N).
    # 1e6 * ~10 ops = 1e7 ops. Python can do ~1e7-1e8 simple ops/sec.
    # So ~0.1 - 0.5 seconds. Very safe.
    
    pp_list = [1] * (N_MAX_VAL + 1)
    
    for i in range(2, N_MAX_VAL + 1):
        x = i
        current_pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                current_pp *= (p ** e)
        pp_list[i] = current_pp
        
    # Convert to numpy array for fast product calculation?
    # Or just use the list.
    # We need max pp_list[n] * pp_list[n+1] for n in [10000, 999999].
    
    best_n = N_MIN
    best_score = 0
    
    # Loop through the range
    # Using a list is fast enough.
    for n in range(N_MIN, N_MAX + 1):
        score = pp_list[n] * pp_list[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
        # Time check
        if time.time() > deadline - 5:
            break
            
    return best_n

# EVOLVE-BLOCK-END
