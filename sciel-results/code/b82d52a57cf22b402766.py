"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255
    such that gcd(a, b) > 1 for all distinct pairs a, b.
    
    The optimal strategy is to take all multiples of the smallest prime factor of 255.
    255 = 3 * 5 * 17.
    Smallest prime factor is 3.
    The set of multiples of 3 in [1, 255] has size floor(255/3) = 85.
    Any two elements in this set share the factor 3, so their gcd is at least 3.
    This set contains 255.
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
