import itertools

def construct():
    """
    Returns exactly 12 distinct positive integers <= 1048576 such that every subset
    (including the empty subset) has a distinct sum. Minimizes the largest integer.
    
    The optimal solution for minimizing the maximum element in a set of n distinct
    positive integers with distinct subset sums is the set of powers of 2:
    {1, 2, 4, ..., 2^(n-1)}.
    
    This is because the subset sums of powers of 2 form the contiguous range [0, 2^n - 1],
    which is the densest possible packing of 2^n distinct sums. Any deviation creates
    holes in the sum set, which forces subsequent elements to be larger to maintain
    distinctness.
    
    For n=12, the largest element is 2^11 = 2048.
    """
    n = 12
    # Powers of 2 from 2^0 to 2^11
    result = [1 << i for i in range(n)]
    
    # Verify: all subset sums must be distinct
    # For powers of 2, subset sums range from 0 to 2^12 - 1 = 4095, all distinct
    # Quick verification using bit properties
    # Each subset sum corresponds to a unique bitmask, so all sums are distinct by construction
    
    # Final safety check: ensure all elements are distinct, positive, and <= maximum
    assert len(result) == n
    assert all(x > 0 for x in result)
    assert all(x <= 1048576 for x in result)
    assert len(set(result)) == n
    
    # Verify distinct subset sums
    subset_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        subset_sums.add(s)
    
    assert len(subset_sums) == 1 << n, "Subset sums are not all distinct!"
    
    return result
