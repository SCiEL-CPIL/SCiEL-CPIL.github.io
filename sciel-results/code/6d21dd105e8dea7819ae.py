"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the maximum-cardinality subset of [1, 255] containing 255
    such that gcd(a, b) > 1 for all distinct pairs.
    
    Since 255 = 3 * 5 * 17, any element must share a prime factor with 255,
    meaning it must be divisible by 3, 5, or 17.
    
    The set of all multiples of the smallest prime factor (3) forms a valid
    pairwise non-coprime set of size floor(255/3) = 85.
    
    This is provably optimal: mixing multiples of 5 or 17 that are not
    divisible by 3 would require removing multiples of 3 that don't share
    those factors, resulting in a net loss in cardinality.
    """
    # Build the set of all multiples of 3 in [1, 255]
    # This guarantees gcd(a,b) >= 3 for any distinct pair a, b in the set
    # and includes 255 since 255 % 3 == 0
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
