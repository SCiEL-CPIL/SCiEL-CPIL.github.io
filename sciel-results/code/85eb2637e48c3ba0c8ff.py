def construct():
    # The maximum number of distinct integers in 2..40 with distinct subset products
    # is bounded by the number of primes <= 40, which is 12.
    # The primes themselves form a valid set because their exponent vectors
    # are the standard basis vectors in the space of prime exponents.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return primes
