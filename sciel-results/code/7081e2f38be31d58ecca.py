"""Rewritten program for the suite task; uses direct search with trial division."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    LIMIT = 1000000
    
    # Helper function to compute Euler's totient phi(n)
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

    # Iterate n from 2 upwards
    # We can skip 1 as phi(1)=1 which is not div by 120
    for n in range(2, LIMIT + 1):
        # Time check occasionally to avoid hanging if something goes wrong
        if time.time() > deadline:
            # Fallback to a known valid large number if we somehow didn't find a small one
            # phi(14400) = 3840, 3840 % 120 == 0
            return 14400
        
        val = phi(n)
        if val % 120 == 0:
            return n
            
    # If loop finishes without finding one (should not happen for limit 1e6)
    return 14400
# EVOLVE-BLOCK-END
