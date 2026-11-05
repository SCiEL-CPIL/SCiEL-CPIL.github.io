"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The optimal solution for minimizing the largest element in a set of n
    integers with distinct subset sums is the set of powers of 2:
    {1, 2, 4, ..., 2^(n-1)}.
    
    For n=12, the largest element is 2^11 = 2048.
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
