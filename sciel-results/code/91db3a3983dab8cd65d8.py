"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    Uses the arithmetic progression {226, 227, ..., 256} of size 31.
    This set satisfies the condition that the minimum sum of k elements is
    strictly greater than the maximum sum of k-1 elements for all k.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
