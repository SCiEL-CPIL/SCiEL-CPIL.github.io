"""
Deterministic solution for Large sets with distinct subset products.
The maximum size of a multiplicatively independent set of integers in [2, 40]
is equal to the number of primes in that range, because the exponent vectors
of the integers form a linearly independent set in Z^k, where k is the number
of primes. Any set of size > k would be linearly dependent, implying a collision
in subset products.

Primes in [2, 40]:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
Count: 12

This program returns these 12 primes.
"""

def construct():
    # Primes in range 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Optional: Verification (not strictly necessary for speed, but good for sanity)
    # The set of primes is always multiplicatively independent.
    # We just return them.
    
    return primes
