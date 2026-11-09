"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # Primes >= 11
    primes = [11, 13, 17, 19, 23, 29, 31, 37]
    # Composites with prime factors in {2, 3, 5, 7}
    # Chosen to have distinct subset sums in exponent space
    composites = [2, 4, 16, 3, 9, 5, 25, 7]
    
    result = primes + composites
    # Ensure types are correct
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
