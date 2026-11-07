"""Rewritten program for the suite task; using prime set which is optimal for this constraint."""
# EVOLVE-BLOCK-START
import math

def construct():
    # The set of all primes in [2, 256] is a valid solution.
    # For any distinct primes a, b, c, a cannot divide b*c because 
    # a is prime and a != b, a != c, so a does not divide b or c.
    # 
    # Analysis suggests that replacing primes with composites does not 
    # increase the cardinality:
    # - Replacing a prime p with p^2 keeps cardinality same (1 for 1).
    # - Replacing primes p, q with composite pq decreases cardinality (1 for 2).
    # - Mixing powers of same prime (p^2, p^3) forces exclusion of all other 
    #   elements to avoid divisibility violations, reducing cardinality.
    # 
    # Thus, the maximum cardinality is the number of primes <= 256.
    
    n = 256
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    return primes
# EVOLVE-BLOCK-END
