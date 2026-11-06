"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent
    coprime pairs in its sorted divisors.
    
    Strategy:
    The number of coprime adjacent pairs is maximized by numbers with many
    distinct prime factors (square-free numbers). This is because divisors
    of square-free numbers are subsets of the prime factors, and the sorted
    order tends to alternate between including and excluding small primes
    in a way that creates many disjoint (coprime) adjacent sets.
    In contrast, numbers with high exponents on small primes (like powers of 2)
    create long runs of divisors sharing those small primes, reducing the
    count of coprime pairs.
    
    We maximize the number of distinct prime factors k such that the product
    of the first k primes is <= 10^18.
    The product of the first 15 primes (up to 47) is ~6.15 * 10^17, which is
    within the limit. The product of the first 16 primes (up to 53) exceeds
    10^18.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
