# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Theoretical Minimum:
    Let the set be {a_1, ..., a_n} with a_1 < ... < a_n.
    For all subset sums to be distinct, the set of sums S_{n-1} from the first n-1 elements
    and the set S_{n-1} + a_n must be disjoint.
    This implies a_n > sum(a_1, ..., a_{n-1}).
    To minimize a_n, we minimize the previous elements.
    a_1 >= 1
    a_2 > a_1 >= 1 => a_2 >= 2
    a_3 > a_1 + a_2 >= 3 => a_3 >= 4
    ...
    a_k >= 2^(k-1)
    Thus a_12 >= 2^11 = 2048.
    The set {1, 2, 4, ..., 2048} achieves this bound.
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
