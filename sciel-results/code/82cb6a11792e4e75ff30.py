"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

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

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The answer is 143: phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
    We verify this is minimal by checking all n < 143.
    """
    deadline = time.time() + 50  # Safety margin within 60s limit
    
    # Verify 143 is correct and minimal
    target = 120
    best = None
    
    # Check candidate 143 first
    if _phi(143) % target == 0:
        best = 143
    
    # Verify no smaller n works (for robustness)
    for n in range(1, 143):
        if time.time() > deadline:
            break
        if _phi(n) % target == 0:
            best = n
            break
    
    # If for some reason verification didn't complete, return 143 as known optimal
    if best is None:
        best = 143
    
    return best
# EVOLVE-BLOCK-END
