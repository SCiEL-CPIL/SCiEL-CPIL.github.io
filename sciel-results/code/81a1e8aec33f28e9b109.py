def construct():
    """
    Returns a list of n=12 distinct positive integers such that all 2^n subset sums
    are distinct, minimizing the largest integer.
    
    The optimal solution is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    
    Proof:
    Let the sorted integers be a_1 < a_2 < ... < a_n.
    Let S_{k} = sum(a_1...a_k).
    For all subset sums to be unique, the set of sums using {a_1...a_{k-1}} and the set
    of sums using {a_1...a_{k-1}} plus a_k must be disjoint.
    The sums using the first k-1 elements lie in [0, S_{k-1}].
    The sums using the first k elements include a_k + s for s in [0, S_{k-1}], which lie in [a_k, a_k + S_{k-1}].
    For disjointness, we need a_k > S_{k-1}.
    Since a_k is an integer, a_k >= S_{k-1} + 1.
    
    To minimize a_n, we minimize each a_k:
    a_1 >= 1 -> a_1 = 1. S_1 = 1.
    a_2 >= S_1 + 1 = 2 -> a_2 = 2. S_2 = 3.
    a_3 >= S_2 + 1 = 4 -> a_3 = 4. S_3 = 7.
    ...
    a_k >= 2^(k-1).
    
    Thus the minimal largest element for n=12 is 2^11 = 2048.
    """
    n = 12
    # Powers of 2 from 2^0 to 2^(n-1)
    return [1 << i for i in range(n)]
