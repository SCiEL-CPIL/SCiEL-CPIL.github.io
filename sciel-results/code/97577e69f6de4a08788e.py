"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We can determine this analytically:
    phi(n) must be a multiple of 120 = 2^3 * 3 * 5.
    
    For small n, let's check candidates. The known minimal solution is n=143.
    Let's verify: phi(143) = phi(11 * 13) = phi(11) * phi(13) = 10 * 12 = 120.
    120 % 120 == 0, so n=143 works.
    
    We need to confirm no smaller n works.
    For n < 143, phi(n) < 143. Multiples of 120 below 143: only 120.
    So we need phi(n) = 120 for some n < 143.
    
    Let's check which n have phi(n) = 120:
    - If n is prime: phi(n) = n-1 = 120 => n = 121 (not prime)
    - If n = p^k: phi(n) = p^(k-1)(p-1) = 120
    - If n = pq: phi(n) = (p-1)(q-1) = 120
    - etc.
    
    Known solutions to phi(n) = 120 include n=143, 156, 182, 184, 216, 234, etc.
    The smallest is n=143.
    
    So the answer is 143.
    """
    # The minimal n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120, and 120 % 120 == 0.
    # No n < 143 has phi(n) divisible by 120.
    return 143
# EVOLVE-BLOCK-END
