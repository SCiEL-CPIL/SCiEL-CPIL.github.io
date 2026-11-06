"""Rewritten program to maximize coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
def construct():
    # The product of the first 15 primes (primorial 47#) is 614889782588491410,
    # which is <= 10^18.
    # The product of the first 16 primes is > 10^18.
    # Using the maximum number of distinct prime factors (squarefree) maximizes
    # the expected number of coprime adjacent pairs due to the high probability
    # of coprimality for squarefree numbers (0.75 per prime) and the large
    # number of divisors (2^15).
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
