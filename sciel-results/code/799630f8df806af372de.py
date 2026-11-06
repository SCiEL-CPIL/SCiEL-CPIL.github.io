# EVOLVE-BLOCK-START
import time

def compute_phi(n):
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
    # We need to find the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # Since we want the smallest n, we can iterate from 1 upwards.
    # The answer is expected to be small, so this is much faster than a full sieve.
    
    for n in range(1, 1000001):
        if compute_phi(n) % 120 == 0:
            return n
            
    # Fallback (should not be reached)
    return 14400
# EVOLVE-BLOCK-END
