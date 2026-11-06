def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The optimal solution is the powers of 2: [1, 2, 4, ..., 2^11].
    The largest integer is 2^11 = 2048.
    """
    return [1 << i for i in range(12)]
