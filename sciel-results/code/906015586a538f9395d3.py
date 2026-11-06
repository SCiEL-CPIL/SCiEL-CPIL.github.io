# EVOLVE-BLOCK-START
def construct():
    """
    Construct exactly 12 distinct positive Python integers <= 1048576
    such that every subset has a different sum. Minimize the largest integer.
    
    The minimal largest integer for n distinct positive integers with distinct subset sums
    is 2^(n-1). This is achieved by the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
