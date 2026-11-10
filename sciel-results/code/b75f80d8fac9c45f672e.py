"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of 171 distinct integers in 1..256 such that 
    subset sums with different cardinalities are distinct.
    
    The set is the arithmetic progression {86, 87, ..., 256}.
    This satisfies min_sum(k) > max_sum(k-1) for all k=1..171.
    """
    return list(range(86, 257))

# EVOLVE-BLOCK-END
