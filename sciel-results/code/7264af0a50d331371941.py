def construct():
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    n = 1
    for p in primes:
        n *= p
    n *= (2 ** 6)
    n *= (3 ** 5)
    return n
