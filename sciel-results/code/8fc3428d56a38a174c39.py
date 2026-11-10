"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a set of 12 distinct integers in [0, 128] that minimizes the size
    of the largest Sidon subset.
    
    Strategy:
    To minimize the largest Sidon subset, we want to maximize the number of
    additive collisions (a+b = c+d) within the set. The most effective way to
    do this is to compress the elements into the smallest possible range, as
    this increases the density of sums and the likelihood of collisions.
    
    The contiguous interval [0, 11] is the most compressed set of 12 distinct
    integers. It is analytically likely to have a smaller maximum Sidon subset
    than any spread-out or clustered set.
    """
    return list(range(12))
# EVOLVE-BLOCK-END
