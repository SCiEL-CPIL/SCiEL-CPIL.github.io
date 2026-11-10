import time
import math
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s kill switch
    
    N = 1000000
    best_n = 10000
    best_prod = 0
    
    # Generate primes up to sqrt(N+1)
    limit = int(math.isqrt(N + 1))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    primes = [int(p) for p in np.where(is_prime)[0]]
    
    # Generate powerful numbers up to N+1
    # A powerful number is a product of prime powers p^e with e >= 2
    powerful_nums = set()
    powerful_nums.add(1)  # 1 is technically powerful (empty product)
    
    # For each prime, generate all prime powers p^e <= N+1 where e >= 2
    prime_powers = {}
    for p in primes:
        pe = p * p
        while pe <= N + 1:
            prime_powers.setdefault(p, []).append(pe)
            pe *= p
    
    # Generate all powerful numbers by combining prime powers with disjoint supports
    # Start with just single prime powers, then combine
    # For efficiency, we'll build up incrementally
    
    # First, collect all single prime powers
    single_pp = list(prime_powers.keys())
    all_pps = []
    for p in single_pp:
        for pe in prime_powers[p]:
            all_pps.append((p, pe))
    
    # Build powerful numbers by combining
    # Use a dictionary: powerful_num -> set of primes used
    powerful_map = {1: set()}
    
    # Add single prime powers
    for p, pe in all_pps:
        powerful_map[pe] = {p}
    
    # Now combine: for each existing powerful number, try multiplying by new prime powers
    # with disjoint prime supports
    # This can explode, so we need to be careful
    
    # Alternative approach: directly search for consecutive pairs
    # n = A * S1, n+1 = B * S2 where A, B are powerful with disjoint supports
    # and S1, S2 are squarefree
    
    # Let's use a different strategy: enumerate powerful numbers and check consecutive pairs
    
    # Generate all powerful numbers up to N+1
    # Use recursive generation
    def gen_powerful(idx, current, used_primes):
        if current > N + 1:
            return
        if current >= 10000 and current <= N + 1:
            powerful_nums.add(current)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        # Option 1: don't use this prime
        gen_powerful(idx + 1, current, used_primes)
        
        # Option 2: use p^e for e >= 2
        pe = p * p
        while pe <= N + 1 and current * pe <= N + 1:
            gen_powerful(idx + 1, current * pe, used_primes | {p})
            pe *= p
    
    gen_powerful(0, 1, set())
    
    # Now for each powerful number A, find B such that A*S1 and B*S2 are consecutive
    # A*S1 + 1 = B*S2 => B*S2 - A*S1 = 1
    # For small squarefree S1, S2
    
    # Precompute squarefree numbers up to some limit
    # S1 and S2 should be small for this to be efficient
    
    # Actually, let's try a more direct approach:
    # For each powerful number A <= N, for each small squarefree S1,
    # compute n = A * S1, check if n is in range, then compute n+1,
    # factor n+1 to get its powerful part B, and check if B * S2 = n+1
    # for some small squarefree S2
    
    # This is still complex. Let me try the enumeration approach more carefully.
    
    # Generate powerful numbers
    powerful_list = sorted(powerful_nums)
    
    # For each powerful number A, try small squarefree multipliers
    # S1 is squarefree, so S1 = product of distinct primes not in A's support
    
    # Precompute small primes for squarefree numbers
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Generate small squarefree numbers (products of subsets of small_primes)
    def gen_squarefree(idx, current):
        if current > 1000:  # Limit S1, S2 to be small
            return
        yield current
        if idx >= len(small_primes):
            return
        for i in range(idx, len(small_primes)):
            if current * small_primes[i] <= 1000:
                yield from gen_squarefree(i + 1, current * small_primes[i])
    
    squarefree_nums = list(gen_squarefree(0, 1))
    
    # For each powerful number A, for each squarefree S1:
    # n = A * S1
    # Check if 10000 <= n <= N
    # Compute n+1, factor it, get powerful part B
    # Check if B * S2 = n+1 for some squarefree S2
    
    # To make this efficient, precompute the powerful part for all numbers
    # But that's what we're trying to avoid...
    
    # Alternative: for each powerful number A, enumerate possible n values
    # and check if n+1 has a large powerful part
    
    # Let's try a hybrid: compute powerful parts using a sieve, but only for
    # numbers that are likely to be candidates
    
    # Actually, let's just do the full sieve but optimize it
    
    pp = np.ones(N + 2, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > N + 1:
            continue
        
        # For each multiple m of p^2, compute the exact power of p in m
        max_k = (N + 1) // p2
        for k in range(1, max_k + 1):
            m = k * p2
            if m > N + 1:
                break
            # Compute exponent of p in k
            temp = k
            e = 2
            while temp % p == 0:
                temp //= p
                e += 1
            pp[m] *= p**e
    
    # Find the max product
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n+1]
        if prod > best_prod:
            best_prod = prod
            best_n = n
        
        if time.time() > deadline:
            break
    
    return int(best_n)
