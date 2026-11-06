"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Uses number-theoretic analysis: for phi(n) to be divisible by 120 = 2^3 * 3 * 5,
    we need the product of (p_i - 1) terms (times powers of p_i for repeated factors)
    to cover these factors.
    
    Key insight: primes p where p-1 divides 120 are: 2, 3, 5, 7, 11, 13, 31.
    But we also need primes where p-1 contributes factors that combine to give 120.
    
    The minimal n is 143 = 11 * 13, since phi(143) = 10 * 12 = 120.
    """
    # Direct computation: verify 143 is the answer
    # phi(143) = phi(11*13) = phi(11)*phi(13) = 10*12 = 120
    # 120 % 120 == 0, so 143 works
    
    # Verify minimality by checking all n < 143
    # This is fast enough (only 142 checks)
    for n in range(1, 143):
        if _phi(n) % 120 == 0:
            return n
    
    return 143

def _phi(n):
    """Compute Euler's totient phi(n) via trial division."""
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
# EVOLVE-BLOCK-END
