"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of 31 distinct integers in 1..256 such that
    subset sums with different cardinalities are distinct.
    
    The set {226, 227, ..., 256} is an arithmetic progression that satisfies
    the condition min_sum(k) > max_sum(k-1) for all k.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
