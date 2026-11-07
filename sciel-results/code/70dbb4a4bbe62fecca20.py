"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

N = 256

def _prime_factors_up_to(n):
    """Compute prime factors for each number up to n using sieve."""
    factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not factors[i]:  # i is prime
            for j in range(i, n + 1, i):
                factors[j].append(i)
    return factors

def construct():
    deadline = time.time() + 85
    
    factors = _prime_factors_up_to(N)
    
    # Greedy: iterate from largest to smallest, pick if no prime factor conflict
    used_primes = set()
    result = []
    
    # 1 is coprime with everything, include it
    result.append(1)
    
    # Process numbers from largest to smallest
    for m in range(N, 1, -1):
        conflict = False
        for p in factors[m]:
            if p in used_primes:
                conflict = True
                break
        if not conflict:
            result.append(m)
            for p in factors[m]:
                used_primes.add(p)
    
    # Return as list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
