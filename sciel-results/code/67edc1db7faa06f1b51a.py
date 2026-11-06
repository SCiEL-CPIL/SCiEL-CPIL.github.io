"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the largest known subset of [1,255] containing 255
    such that gcd(a,b) > 1 for all distinct pairs a, b in the set.
    
    Strategy: 255 = 3 * 5 * 17. The smallest prime factor is 3.
    Taking all multiples of 3 in [1, 255] gives a set of size floor(255/3) = 85.
    Every pair in this set has gcd >= 3 > 1, and 255 is included.
    
    This is optimal by the theorem that for n with smallest prime factor p,
    the largest pairwise non-coprime subset of [1,n] containing n has size floor(n/p).
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
