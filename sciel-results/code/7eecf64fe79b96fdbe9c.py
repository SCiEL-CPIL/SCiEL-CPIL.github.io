"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _phi(n):
    """Compute Euler's totient function for n."""
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

def construct():
    # We search for the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    # Since the minimum n is expected to be small, we iterate from 1 upwards.
    # A sieve is also an option, but on-the-fly calculation is sufficient for small n.
    # However, to be robust and fast, we can precompute phi up to a limit if needed, 
    # but simple iteration with a fast phi function is very effective here.
    
    limit = 1000000
    for n in range(1, limit + 1):
        if _phi(n) % 120 == 0:
            return n
    return -1
# EVOLVE-BLOCK-END
