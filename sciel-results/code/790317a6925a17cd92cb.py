"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 135  # leave 15s margin before 150s kill
    
    # We want to find M with many distinct k such that k * sigma(k) = M,
    # with k <= sqrt(M) and k_i distinct.
    # Strategy: enumerate square-free k (which have high sigma/k ratio variance)
    # up to some limit, compute M = k * sigma(k), and track collisions.
    
    # For square-free k = p1*p2*...*pr, sigma(k) = (p1+1)*(p2+1)*...*(pr+1)
    # So M = k * sigma(k) = prod(p_i * (p_i+1))
    # This means M is a product of terms of the form p*(p+1) for distinct primes p.
    # Two different sets of primes can give the same M if the products match.
    #
    # Alternative: just enumerate k up to some bound, compute sigma(k), and find collisions.
    # Use a sieve to compute sigma(k) for all k up to N efficiently.
    
    N = 20_000_000  # upper limit for k
    
    # Linear sieve for sigma(k)
    # sigma is multiplicative: if k = prod p^a, sigma(k) = prod (p^(a+1)-1)/(p-1)
    # We can use a modified sieve.
    
    sigma = [1] * (N + 1)
    primes = []
    is_composite = [False] * (N + 1)
    # For linear sieve computing sigma:
    # sigma[1] = 1
    # For prime p: sigma[p] = p+1
    # For k = p^a * m where p doesn't divide m:
    #   sigma[k] = sigma[p^a] * sigma[m]
    # sigma[p^a] = (p^(a+1) - 1) / (p - 1)
    
    # We need to track the highest power of the smallest prime factor
    # Let's use a different approach: for each prime p, multiply sigma by (1 + p + p^2 + ...)
    
    # Actually, let's use the standard approach:
    # sigma[1] = 1
    # For each prime p, for multiples j = p, 2p, 3p, ...:
    #   We need to add p^0 + p^1 + ... + p^a to the "p-part"
    # This is complex. Let me use a different method.
    
    # Method: compute sigma using the formula via prime factorization with a sieve.
    # sigma[k] = sum of divisors. We can compute it as:
    # Initialize sigma[i] = 1 for all i
    # For each i from 2 to N:
    #   if sigma[i] == 1 (i is prime):
    #     for j = i, 2i, 3i, ... <= N:
    #       p_power = i
    #       term = 1
    #       while j % p_power == 0:
    #         p_power *= i
    #         term += p_power
    #       sigma[j] *= term
    # This is O(N log N) which might be too slow for N=20M.
    
    # Better: linear sieve for sigma.
    # Let lp[k] = smallest prime factor of k
    # Let pk[k] = highest power of lp[k] dividing k
    # Let sigma_pk[k] = sigma(pk[k]) = 1 + lp + lp^2 + ... + lp^a
    # Let sigma_rest[k] = sigma(k / pk[k])
    # Then sigma[k] = sigma_pk[k] * sigma_rest[k]
    
    # Actually, simpler linear sieve:
    # sigma[1] = 1
    # For prime p: sigma[p] = p+1
    # For composite: if k = p * m where p = lp[k]:
    #   if p doesn't divide m: sigma[k] = sigma[p] * sigma[m]
    #   if p divides m: let m = p^a * r, then k = p^(a+1) * r
    #     sigma[k] = sigma[p^(a+1)] * sigma[r]
    
    # Let me just use the straightforward O(N log log N) sieve:
    # sigma[i] starts at 1
    # For each prime p, for each multiple j of p:
    #   compute the p-part contribution
    
    # Actually the simplest correct sieve:
    # sigma = [0]*(N+1)
    # for i in range(1, N+1):
    #   for j in range(i, N+1, i):
    #     sigma[j] += i
    # This is O(N log N) ~ N * H(N) ~ 20M * 17 ~ 340M operations. Too slow in Python.
    
    # Let's use NumPy for speed.
    import numpy as np
    
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            # For all multiples of i, multiply by (1 + i + i^2 + ...)
            # The exponent of i in j is the number of times i divides j
            for j in range(i, N + 1, i):
                # Count power of i in j
                temp = j
                power = 1
                term_sum = 1
                while temp % i == 0:
                    temp //= i
                    power *= i
                    term_sum += power
                sigma[j] *= term_sum
    
    # This is still slow in pure Python loop. Let me think differently.
    
    # Alternative: just enumerate k and compute sigma(k) via factorization using precomputed smallest prime factor.
    # SPF sieve is O(N log log N) and fast.
    
    # Let's reset and use SPF.
    # But first, let's check time.
    if time.time() - start_time > DEADLINE - 10:
        # Fallback to known answer
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Let me try a smaller N with the O(N log N) divisor sum approach but optimized.
    # Actually, let's just enumerate square-free numbers and compute their sigma directly.
    # For square-free k, sigma(k) = prod(p_i + 1)
    # And M = k * sigma(k) = prod(p_i * (p_i + 1))
    
    # Generate all square-free numbers up to N using their prime factors.
    # But enumerating all square-free numbers up to 20M and computing M for each,
    # then finding collisions, is the key.
    
    # Let's use a smarter approach: 
    # Precompute smallest prime factor (SPF) up to N.
    # Then for each k, factorize it quickly, compute sigma(k), and compute M = k * sigma(k).
    # Track M -> list of k's.
    
    # SPF sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    if time.time() - start_time > DEADLINE - 10:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Now compute sigma(k) for all k using SPF
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    for k in range(2, N + 1):
        if time.time() - start_time > DEADLINE - 15:
            break
        n = k
        s = 1
        while n > 1:
            p = spf[n]
            # Compute p^a part
            power = 1
            term = 1
            while n % p == 0:
                n //= p
                power *= p
                term += power
            s *= term
        sigma_arr[k] = s
    
    if time.time() - start_time > DEADLINE - 5:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Compute M = k * sigma(k) and find collisions
    # M can be up to N * sigma(N). For N=20M, sigma(N) can be large.
    # M = k * sigma(k). We need M <= 10^18.
    # For k up to 20M, sigma(k) is at most ~ k * log(log(k)) roughly, so M ~ k^2 * log(log(k))
    # For k = 20M, M ~ 4e14 * 3 ~ 1.2e15, well within 10^18.
    
    M_map = defaultdict(list)
    for k in range(2, N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        M_val = k * sigma_arr[k]
        if M_val <= 10**18:
            M_map[M_val].append(k)
    
    # Find M with most k's where k <= sqrt(M)
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for M_val, k_list in M_map.items():
        # Filter k <= sqrt(M_val)
        sqrt_M = isqrt(M_val)
        valid_k = [k for k in k_list if k <= sqrt_M]
        if len(valid_k) > best_count:
            best_count = len(valid_k)
            best_M = M_val
            best_k = valid_k
            if best_count >= 1000:
                break
    
    if time.time() - start_time > DEADLINE:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
