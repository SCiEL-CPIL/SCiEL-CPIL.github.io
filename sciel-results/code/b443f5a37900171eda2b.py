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

    # 2. Multiplicative number-theoretic sieve to compute powerful part P(n) for all n in [1, N]
    #    P(n) = product of p^e for each prime p where e = v_p(n) >= 2
    P = [1] * (N + 1)
    
    # Find all primes up to N
    primes = []
    for i in range(2, N + 1):
        if spf[i] == i:
            primes.append(i)
    
    for p in primes:
        pk = p * p
        while pk <= N:
            p_next = pk * p
            # For each multiple m of pk, if m is not a multiple of p_next,
            # then v_p(m) = k, so we multiply p^k = pk into P[m].
            for m in range(pk, N + 1, pk):
                if m % p_next != 0:
                    P[m] *= pk
            pk = p_next
    
    # 3. Find n in [MIN_N, N-1] that maximizes P[n] * P[n+1]
    #    Note: n=N is not evaluated because P[N+1] is out of bounds, 
    #    but P(N)*P(N+1) is small (P(N+1)=1 for N=10^6) and thus not the maximum.
    
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    # Check if we are running out of time (unlikely for this N)
    if time.time() > DEADLINE:
         pass # Return best found so far

    return best_n

# EVOLVE-BLOCK-END
