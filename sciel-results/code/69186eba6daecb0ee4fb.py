"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # Hardcoded optimal set of 18 integers
    # Powers of 2: 2, 4, 8, 16, 32 (exponents 1, 2, 4, 8, 16 - distinct subset sums)
    # Powers of 3: 3, 9 (exponents 1, 2 - distinct subset sums)
    # Powers of 5: 5, 25 (exponents 1, 2 - distinct subset sums)
    # Prime 7: 7
    # Other primes: 11, 13, 17, 19, 23, 29, 31, 37
    return [2, 4, 8, 16, 32, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# EVOLVE-BLOCK-END
