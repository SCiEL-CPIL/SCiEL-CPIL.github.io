"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    N = 123457
    
    # Strategy: Compute the Largest Prime Factor (LPF) for all numbers up to N.
    # Then iterate a from 1 to N//2 and find the pair minimizing max(LPF(a), LPF(N-a)).
    
    # 1. Compute LPF array
    # lpf[i] will hold the largest prime factor of i.
    lpf = [1] * (N + 1)
    
    # Sieve for primes and update LPF
    # We iterate potential primes p. If lpf[p] == p (or initially 1 if we handle 1 separately, but 1 is not prime), 
    # actually, let's just iterate i from 2 to N.
    # If lpf[i] == 1, it means i is prime (since no smaller prime has marked it as composite in a standard sieve, 
    # but here we are marking with the *current* prime p).
    # Better: Standard sieve to find primes, then for each prime p, mark multiples.
    
    # Let's use a boolean array to identify primes first, or just use the LPF array itself.
    # If we initialize lpf with 1, then for each i from 2 to N:
    # If lpf[i] == 1, i is prime.
    # Then for j in range(i, N+1, i): lpf[j] = i
    # Since we go i=2, 3, 4..., when we hit a prime i, we overwrite lpf[j] with i for all multiples j.
    # The last prime that divides j will be the largest one.
    
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Find optimal a
    best_a = 1
    best_score = float('inf')
    
    # We only need to check a up to N//2 because the pair (a, b) is the same as (b, a)
    for a in range(1, N // 2 + 1):
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
