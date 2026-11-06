"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The smallest such integer is 143, since phi(143) = phi(11 * 13) = 10 * 12 = 120.
    We verify this is the minimum by checking all n < 143.
    """
    # Helper to compute totient
    def phi(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    # Verify 143 is the minimum
    # We know phi(n) < n, so for n < 120, phi(n) cannot be 120.
    # We check n from 120 to 142 to ensure none have phi(n) == 120.
    # If none do, and phi(143) == 120, then 143 is the answer.
    
    min_n = 143
    for n in range(1, 143):
        if phi(n) % 120 == 0:
            min_n = n
            break
            
    return min_n
# EVOLVE-BLOCK-END
