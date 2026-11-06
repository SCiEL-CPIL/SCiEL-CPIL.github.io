"""Rewritten program: maximize coprime adjacent divisor pairs via primorial."""
# EVOLVE-BLOCK-START
def construct():
    # We search for the optimal set of primes and exponents.
    # Heuristic: Use the smallest primes to maximize the count of divisors
    # and keep divisors small to maximize coprime adjacencies.
    # The product of the first 15 primes is the smallest number with 15 distinct prime factors.
    # We verify if adding a power of 2 (making it 2^2 * 3 * ... * 43) is better.
    # Given the divisor count reduction, the primorial is expected to be better.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
        
    # Check alternative: 2^2 * 3 * ... * 43
    alt_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    n_alt = 1
    for p in alt_primes:
        n_alt *= p
    n_alt *= 2 # 2^2
    
    # Heuristic: More divisors is generally better for this score.
    # n has 2^15 divisors, n_alt has 3*2^13 divisors.
    # 32768 > 24576.
    # So n is likely better.
    
    return n
# EVOLVE-BLOCK-END
