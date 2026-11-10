import time

def construct():
    # The optimal strategy is to use a square-free number with the maximum
    # number of distinct prime factors (the largest primorial <= 10^18).
    # This creates a divisor lattice where adjacent divisors frequently share
    # no common prime factor, maximizing tau_perp(n).
    #
    # Product of first 15 primes:
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # = 614889782588491410
    #
    # Adding the 16th prime (53) would give 614889782588491410 * 53 > 10^18,
    # so 15 distinct primes is the maximum.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    
    # Verify n <= 10^18
    assert 2 <= n <= 10**18, f"n={n} out of bounds"
    
    return n
