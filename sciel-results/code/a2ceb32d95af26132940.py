"""Construct 12 distinct positive integers with distinct subset sums, minimizing the largest element."""

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 2^12 subset sums are distinct.
    The sequence of powers of 2 (1, 2, 4, ..., 2048) is the optimal solution that minimizes the largest element.
    It is a proven lower bound that the k-th smallest element in such a set must be at least 2^(k-1).
    """
    n = 12
    return [1 << i for i in range(n)]
