"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a linear sieve to precompute the smallest prime factor (SPF) for all numbers
    up to 1,000,000. Then, for each n starting from 1, it computes phi(n) in O(log n)
    using the precomputed SPF and checks if phi(n) % 120 == 0.
    """
    LIMIT = 1000000
    
    # Precompute smallest prime factor (SPF) for all numbers up to LIMIT
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Iterate n from 1 to LIMIT
    for n in range(1, LIMIT + 1):
        # Compute phi(n) using SPF
        original_n = n
        phi_n = n
        while n > 1:
            p = spf[n]
            if phi_n % p == 0:
                phi_n = phi_n // p * (p - 1)
            while n % p == 0:
                n //= p
        
        if phi_n % 120 == 0:
            return original_n
            
    # Fallback if no solution found (should not happen)
    return 14400

# EVOLVE-BLOCK-END
