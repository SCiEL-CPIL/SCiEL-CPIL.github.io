"""Program for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of
    every nonempty consecutive block are distinct. Maximize length.
    
    Approach: Explicit construction using primes.
    A sequence of distinct primes has the property that all consecutive
    block products are distinct. This is because the prime factorization
    of any product uniquely identifies the set of primes involved, and
    since the primes are in a strictly increasing sequence, the set of
    indices is uniquely determined by the primes present.
    
    This guarantees a valid sequence. The length is the number of primes
    in [2, 128], which is 31.
    """
    # Generate primes in [2, 128]
    primes = []
    for x in range(2, 129):
        is_prime = True
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
    
    return primes
# EVOLVE-BLOCK-END
