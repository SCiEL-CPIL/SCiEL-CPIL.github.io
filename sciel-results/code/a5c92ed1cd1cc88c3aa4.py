def construct():
    # Use a highly composite-like number with powers of small primes
    # to maximize the number of divisors and thus the score.
    # n = 2^3 * 3^2 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    n = 2**3 * 3**2 * 5**2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    return n
