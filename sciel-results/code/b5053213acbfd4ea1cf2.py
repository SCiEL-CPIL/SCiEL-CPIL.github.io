"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The smallest such integer is 143, since phi(143) = phi(11*13) = 10*12 = 120.
    We verify this directly and return it.
    """
    # The known minimal solution is 143: phi(143) = 120, which is divisible by 120.
    # We can verify this quickly:
    # 143 = 11 * 13
    # phi(143) = 143 * (1 - 1/11) * (1 - 1/13) = 143 * (10/11) * (12/13) = 143 * 120/143 = 120
    # 120 % 120 == 0, so 143 is valid.
    
    # To be absolutely certain and robust, we do a quick verification:
    def totient(n):
        result = n
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
            p += 1 if p == 2 else 2
        if m > 1:
            result -= result // m
        return result
    
    # Verify 143
    if totient(143) % 120 == 0:
        return 143
    
    # Fallback: search from 1 upward (should not reach here)
    for n in range(1, 1001):
        if totient(n) % 120 == 0:
            return n
    
    # Final fallback (extremely unlikely)
    return 30030
# EVOLVE-BLOCK-END
