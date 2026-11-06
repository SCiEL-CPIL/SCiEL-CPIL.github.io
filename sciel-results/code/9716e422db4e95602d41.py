def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576 such that all 2^12 = 4096
    subset sums (including the empty subset) are distinct. Minimize the largest element.
    
    The optimal solution is the set of powers of 2: {1, 2, 4, ..., 2048}.
    The largest element is 2048.
    This is derived from the requirement that the sequence must be superincreasing
    to ensure distinct subset sums, and minimizing each term greedily leads to powers of 2.
    """
    n = 12
    # The optimal sequence is 1, 2, 4, ..., 2^(n-1)
    # Largest element is 2^(n-1) = 2048
    return [1 << i for i in range(n)]
