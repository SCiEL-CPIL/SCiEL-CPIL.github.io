"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 123457
    # Sieve to compute largest prime factor for all integers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Generate primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Now for each prime, update lpf for its multiples
    # Iterating in increasing order ensures the last assignment is the largest prime factor
    for p in range(2, N + 1):
        if is_prime[p]:
            for m in range(p, N + 1, p):
                lpf[m] = p
    
    # Iterate through all splits a + b = N
    # Due to symmetry, we only need to check a up to N // 2
    best_max_p = float('inf')
    best_pair = (1, N - 1)
    
    # Local variable access for speed
    lpf_local = lpf
    N_local = N
    
    for a in range(1, N_local // 2 + 1):
        b = N_local - a
        pa = lpf_local[a]
        pb = lpf_local[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_pair = (a, b)
            # Theoretical minimum for this N is > 2, so we don't break early on small numbers
            # unless we are certain. But N is fixed, so we just run the loop.
    
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
