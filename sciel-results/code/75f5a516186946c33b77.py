"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    The set is an arithmetic progression {226, 227, ..., 256} of size 31.
    This is the maximum possible size for this constraint.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
