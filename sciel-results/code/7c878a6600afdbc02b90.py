"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The maximum number of multiplicatively independent integers in [2, 40]
    # is equal to the number of primes in that range, because the exponent
    # vectors of these numbers must be linearly independent over Z.
    # The number of primes <= 40 is 12.
    # Any set of 12 such numbers will yield a score of 12/12 = 1.0.
    # We return the primes themselves as they are the simplest valid set.
    
    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure types are Python int and list
    return [int(p) for p in primes]
# EVOLVE-BLOCK-END
