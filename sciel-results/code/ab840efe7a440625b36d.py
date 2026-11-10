import math
import time

def construct():
    # The goal is to maximize the number of adjacent coprime pairs in the sorted divisors of n.
    # For a squarefree integer n = p1 * p2 * ... * pk with primes p1 < p2 < ... < pk,
    # the divisors are all 2^k products of subsets. The number of adjacent coprime pairs
    # is maximized when n has as many prime factors as possible.
    #
    # The product of the first k primes is the primorial p_k#.
    # p_15# = 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410 ≈ 6.15 * 10^17
    # p_16# = p_15# * 53 ≈ 3.26 * 10^19 > 10^18
    #
    # So the maximum number of prime factors for n <= 10^18 is 15.
    # Any non-squarefree n would have fewer distinct prime factors or be larger for the same count.
    # Thus, the optimal n is the product of the first 15 primes.
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    n = 1
    for p in primes:
        n *= p
        # Time check (though this loop is instant, we keep it for structural compliance)
        if time.time() > deadline:
            # Fallback to a smaller valid n if we were somehow slow (unlikely)
            return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    # Safety check
    if n > 10**18:
        # This should not happen for the hardcoded list, but just in case
        # Fallback to seed logic or smaller set
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
        
    return n
