"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Sieve to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # For each prime, find max power p^k <= n
    selected = [1]
    for p in range(2, n + 1):
        if is_prime[p]:
            # Find max power of p that is <= n
            val = p
            while val * p <= n:
                val *= p
            selected.append(val)
    
    # Verify pairwise coprime (safety check)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # This should never happen, but handle gracefully
                pass
    
    # Ensure all are Python ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
