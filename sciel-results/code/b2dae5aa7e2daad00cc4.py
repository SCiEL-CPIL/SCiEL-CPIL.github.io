"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of n=12 distinct positive integers <= maximum=1048576
    such that all subset sums are distinct. Minimize the largest integer.
    
    The theoretical minimum for the largest element a_n in a set of n distinct
    positive integers with distinct subset sums is 2^(n-1). This is achieved
    by the superincreasing sequence of powers of 2: [1, 2, 4, ..., 2^(n-1)].
    
    For n=12, the optimal sequence is [1, 2, 4, ..., 2048].
    Largest integer: 2048.
    All 2^12 = 4096 subset sums are distinct because each sum corresponds
    uniquely to a binary representation of an integer in [0, 2^12 - 1].
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
