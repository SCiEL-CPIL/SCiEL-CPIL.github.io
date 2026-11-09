import time

def construct():
    """
    Construct a set of n=12 distinct positive integers <= maximum=1048576
    such that all subset sums are distinct. Minimize the largest integer.
    
    Theoretical Minimum:
    For a set of n distinct positive integers with distinct subset sums,
    the sequence must be superincreasing: a_k > sum(a_1...a_{k-1}).
    To minimize the largest element a_n, we choose the smallest possible
    values for each a_k.
    a_1 = 1
    a_k = sum(a_1...a_{k-1}) + 1 = 2^(k-1)
    
    For n=12, the optimal sequence is [1, 2, 4, ..., 2048].
    The largest element is 2^11 = 2048.
    This is the global minimum for the largest element.
    """
    n = 12
    max_val = 1048576
    
    # The optimal sequence is powers of 2: 2^0, 2^1, ..., 2^(n-1)
    # This ensures all 2^n subset sums are unique (binary representation).
    # The largest element is 2^(n-1).
    
    result = [1 << i for i in range(n)]
    
    # Verify constraints (optional but good for safety)
    if max(result) > max_val:
        # Fallback: this should never happen for n=12 and max_val=1048576
        # but if it did, we'd need a different strategy. 
        # Since 2048 < 1048576, this block is effectively dead code for the given params.
        pass
        
    return result
