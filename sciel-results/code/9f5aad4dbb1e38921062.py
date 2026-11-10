"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The condition that all 2^k subset products are distinct is equivalent to
    # the prime exponent vectors of the selected integers being linearly independent.
    # The integers are in [2, 40].
    # The primes in this range are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # There are 12 such primes.
    # The dimension of the vector space formed by prime exponents is 12.
    # Therefore, the maximum number of linearly independent vectors is 12.
    # The primes themselves form a basis (unit vectors), so they are linearly independent.
    # Any set of 12 such integers will have distinct subset products.
    # No set of 13 or more integers can satisfy the condition.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify types are Python int
    return [int(p) for p in primes]
# EVOLVE-BLOCK-END
