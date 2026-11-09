"""Improved program: constructs n = product of smallest k primes for tau_perp maximization.

For squarefree n, divisors form a hypercube. Each divisor d corresponds to a subset S of primes.
Adjacent divisors (in sorted order) often differ by a single prime (one bit flip), giving gcd=1.
Score for squarefree n with k primes: 2^k - 2.
Maximize k such that product of first k primes <= 10^18.
"""
# EVOLVE-BLOCK-START
import math

def construct():
    # Product of first k primes <= 10^18
    # First 15 primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47
    # Product = 614889782588491410 (which is < 10^18)
    # Adding 53: 614889782588491410 * 53 = 32589158477190044730 > 10^18
    # So k=15, score = 2^15 - 2 = 32766
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    
    # Verify n <= 10^18
    assert 2 <= n <= 10**18, f"n={n} out of range"
    return n
# EVOLVE-BLOCK-END
