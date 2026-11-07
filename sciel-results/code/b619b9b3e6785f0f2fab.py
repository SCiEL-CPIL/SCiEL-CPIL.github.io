"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    n = 256
    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:  # i is prime
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    used_primes = set()
    result = [1]  # 1 is coprime with everything
    
    # Iterate from largest to smallest
    for m in range(n, 1, -1):
        # Check if any prime factor of m is already used
        conflict = False
        for p in prime_factors[m]:
            if p in used_primes:
                conflict = True
                break
        if not conflict:
            result.append(m)
            for p in prime_factors[m]:
                used_primes.add(p)
                
    return result
# EVOLVE-BLOCK-END
