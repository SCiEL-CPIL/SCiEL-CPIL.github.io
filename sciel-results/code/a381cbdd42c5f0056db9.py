"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a set of 31 distinct integers in [1, 256] such that subset sums
    of different cardinalities are disjoint.
    
    The set {226, 227, ..., 256} works.
    For any k, the minimum sum of k elements is strictly greater than the
    maximum sum of k-1 elements.
    """
    n = 31
    start = 225
    return [start + i for i in range(1, n + 1)]
# EVOLVE-BLOCK-END
