def construct():
    """
    Returns the product of the first 15 primes.
    This maximizes the number of distinct prime factors within the 10^18 limit,
    which structurally maximizes the number of coprime adjacent divisor pairs.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
