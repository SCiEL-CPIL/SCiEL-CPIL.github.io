"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}
DEADLINE = 90.0

def construct():
    """
    Returns a list of distinct integers in [2, 40] such that all subset products are distinct.
    This is equivalent to selecting the maximum number of integers whose prime exponent vectors
    are linearly independent over Q. The dimension of the space is the number of primes <= 40,
    which is 12. The set of primes themselves forms a linearly independent set of size 12.
    """
    start_time = time.time()
    
    # Generate primes up to P['n']
    # These have exponent vectors that are unit vectors in the basis of primes,
    # hence they are linearly independent.
    n = P['n']
    primes = []
    for a in range(2, n + 1):
        is_prime = True
        # Simple primality test
        if a < 2:
            is_prime = False
        else:
            for d in range(2, int(a**0.5) + 1):
                if a % d == 0:
                    is_prime = False
                    break
        if is_prime:
            primes.append(a)
            
    # The number of primes is the maximum possible size of such a set.
    # Any composite number's vector is a linear combination of prime vectors.
    # Thus, we cannot exceed the number of primes.
    
    # Safety check against deadline (though this runs in microseconds)
    if time.time() - start_time > DEADLINE * 0.9:
        return primes[:12] # Fallback, though unlikely to trigger

    return primes
# EVOLVE-BLOCK-END
