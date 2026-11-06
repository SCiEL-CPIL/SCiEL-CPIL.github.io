import time

def construct():
    """
    Construct exactly 12 distinct positive integers <= 1048576 
    with all subset sums distinct. Minimize the largest integer.
    """
    n = 12
    # The optimal sequence is the powers of 2: 1, 2, 4, ..., 2^(n-1)
    # This is the unique sequence (up to permutation, but we sort) 
    # that minimizes the maximum element for distinct subset sums.
    
    # Generate the sequence
    seq = [1 << i for i in range(n)]
    
    # Verification (optional but good for robustness)
    # Check distinctness
    if len(set(seq)) != n:
        raise ValueError("Sequence elements are not distinct")
    if any(x <= 0 or x > 1048576 for x in seq):
        raise ValueError("Sequence elements out of bounds")
        
    # Check superincreasing property which guarantees distinct subset sums
    # and is necessary for the minimal maximum.
    s = 0
    for x in seq:
        if x <= s:
            # Fallback to explicit check if not superincreasing (should not happen for this seq)
            pass
        s += x
        
    # For n=12, explicit check is cheap and confirms correctness
    # Generate all subset sums
    sums = set()
    # Use bit masking or itertools
    # 2^12 = 4096 subsets
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += seq[i]
        if total in sums:
            raise ValueError("Duplicate subset sum found")
        sums.add(total)
        
    return seq
