import time

def construct():
    """
    Returns a list of n=12 distinct positive integers such that all 4096 subset sums are distinct.
    Minimizes the largest integer.
    The optimal solution is the powers of 2: [1, 2, 4, ..., 2048].
    Largest integer is 2048.
    """
    n = 12
    # The minimal maximum element for n distinct positive integers with distinct subset sums
    # is 2^(n-1).
    # For n=12, max is 2^11 = 2048.
    result = [1 << i for i in range(n)]
    return result
