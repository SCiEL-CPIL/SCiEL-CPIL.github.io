"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs, maximizing cardinality.
    
    The strategy is to select all multiples of the smallest prime factor of 255, which is 3.
    Any two multiples of 3 have a gcd of at least 3.
    255 is a multiple of 3, so it is included.
    The size of this set is floor(255 / 3) = 85.
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
