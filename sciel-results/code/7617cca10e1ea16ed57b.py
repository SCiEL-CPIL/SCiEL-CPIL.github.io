"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78  # safety margin before 90s kill
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Multiplicative sieve to compute powerful part P(n) for all n in [1, N]
    #    P(n) = product of p^e for each prime p where v_p(n) = e >= 2
    #    Initialize P(n) = 1. For each prime p, for each k >= 2 with p^k <= N,
    #    multiply p^k into P(m) for every multiple m of p^k.
    #    Wait, that's not quite right. We need the FULL prime power p^e where e = v_p(n).
    #    Better approach: for each n, factor using SPF and compute powerful part directly.
    #    But that's O(N log N) which might be slow. Let's use a sieve-based approach.
    
    # Actually, let's compute powerful_part for all n using a sieve:
    # For each n from 1 to N, powerful_part[n] = product of p^e where p^e || n and e >= 2.
    # We can do this by: for each prime p, for each exponent e >= 2, for each multiple of p^e,
    # we need to check if v_p(m) == e (i.e., p^e divides m but p^(e+1) doesn't).
    # This is complex. Let's just use SPF-based factorization with caching.
    
    # Precompute powerful_part for all n using SPF - this is O(N * log(N)) but with small constants
    pp = [1] * (N + 1)
    for n in range(2, N + 1):
        if spf[n] == n:  # n is prime
            pp[n] = 1  # prime has no powerful part
        else:
            p = spf[n]
            m = n
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            rest = pp[m] if m <= N else 1
            if e >= 2:
                pp[n] = (p ** e) * rest
            else:
                pp[n] = rest

    # 3. Now scan all consecutive pairs to find max pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = 0

    for n in range(MIN_N, N):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    # Ensure validity
    if best_n < MIN_N or best_n > N - 1:
        best_n = MIN_N

    return best_n

# EVOLVE-BLOCK-END
