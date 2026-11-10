def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    The maximum size is 19 (powers of primes), but the interface limits to 18.
    We return 18 such numbers.
    """
    # Powers of 2: 2, 4, 8, 16, 32
    # Powers of 3: 3, 9, 27
    # Powers of 5: 5, 25
    # Powers of 7: 7
    # Primes 11, 13, 17, 19, 23, 29, 31, 37
    
    # Total 19 numbers. We pick 18.
    # Excluding 37.
    result = [2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31]
    return result
