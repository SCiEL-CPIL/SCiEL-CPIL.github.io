"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    LO, HI = 10000, 1000000
    
    # Compute largest prime factor for all numbers in [LO, HI+1] using a sieve-like approach
    # We only need numbers from LO to HI+1
    # Use a modified sieve: for each prime p, mark multiples of p with p as a factor
    # Then LPF[n] = max prime factor of n
    
    max_val = HI + 1  # we need up to HI+1
    # Initialize LPF array
    lpf = [1] * (max_val + 1)
    
    # Sieve: for each prime, update LPF for its multiples
    # First, find all primes up to sqrt(max_val) using simple sieve
    limit = int(math.isqrt(max_val)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, mark its multiples
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            lpf[multiple] = max(lpf[multiple], p)
    
    # For numbers that are prime themselves (> sqrt(max_val)), they haven't been marked
    # We need to handle those
    # Actually, for any composite number, its LPF is at most itself, and if it's prime, LPF is itself
    # The sieve above marks all composites with their prime factors
    # But for primes > limit, they remain as 1 in lpf array
    # We need to fix this: for each number, if lpf[n] == 1 and n > 1, then n is prime, so lpf[n] = n
    
    # Actually, let's be more careful. The sieve marks p as a factor of all multiples of p.
    # So for a prime p > limit, p itself is a multiple of p, so lpf[p] should be set to p.
    # But we only iterate primes up to limit. So primes > limit are not processed.
    # Let's extend: for primes > limit, we still need to mark them.
    
    # Better approach: iterate all numbers from 2 to max_val, and for each, if it's prime, mark its multiples
    # But that's O(n log log n) which is fine for 1e6
    
    # Let me redo this more carefully
    lpf = [1] * (max_val + 1)
    for p in range(2, max_val + 1):
        if lpf[p] == 1:  # p is prime
            for multiple in range(p, max_val + 1, p):
                lpf[multiple] = max(lpf[multiple], p)
    
    # Now lpf[n] gives the largest prime factor of n for all n in [2, max_val]
    # lpf[1] = 1
    
    # Find the best n in [LO, HI] minimizing max(lpf[n], lpf[n+1])
    best_n = LO
    best_val = max(lpf[LO], lpf[LO + 1])
    
    for n in range(LO, HI):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline - 5:
            break
    
    return best_n
# EVOLVE-BLOCK-END
