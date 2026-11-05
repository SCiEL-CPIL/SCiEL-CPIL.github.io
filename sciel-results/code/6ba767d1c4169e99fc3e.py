import time

def construct():
    """
    Returns a set of n=12 distinct positive integers such that all subset sums are distinct.
    The optimal largest element is 2^(n-1) = 2048.
    The set {1, 2, 4, ..., 2048} achieves this.
    """
    n = 12
    # The optimal set is powers of 2
    return [1 << i for i in range(n)]
