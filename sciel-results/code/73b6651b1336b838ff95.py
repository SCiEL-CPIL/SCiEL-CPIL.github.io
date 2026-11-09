"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the largest subset of [1, 255] containing 255 with gcd(a,b) > 1 for all pairs.
    
    255 = 3 * 5 * 17. Any element must share a factor with 255, so it must be
    divisible by 3, 5, or 17.
    
    The set of multiples of 3 in [1, 255] has size 85. Any two multiples of 3
    share the factor 3, so this is a valid clique.
    
    We verify this is optimal: to add a number divisible by 5 (or 17) but not 3,
    we'd need to remove all multiples of 3 not divisible by 5 (or 17). The trade-off
    is unfavorable: removing ~68 numbers to add 1 is a net loss.
    
    Therefore the maximum cardinality is 85, achieved by all multiples of 3.
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
