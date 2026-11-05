import time

def _validate(arr):
    """Validate: 12 distinct positive ints <= max, all subset sums distinct."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    # Check distinct subset sums
    # For n=12, 4096 sums. We can use a set to check uniqueness.
    # Optimization: if the sequence is superincreasing, it's guaranteed.
    # But for robustness, we do a quick check.
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def construct():
    """
    Construct a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    Theoretical minimum for the largest element $a_n$ in a superincreasing sequence
    of length $n$ starting with $a_1 \ge 1$ is $2^{n-1}$.
    This is achieved by the powers of 2: $1, 2, 4, \dots, 2^{n-1}$.
    
    This implementation constructs the minimal superincreasing sequence directly.
    """
    n = 12
    # Construct the minimal superincreasing sequence
    # a_i >= sum(a_1...a_{i-1}) + 1
    # To minimize a_n, we choose the smallest possible a_i at each step.
    seq = []
    current_sum = 0
    for i in range(n):
        if i == 0:
            next_val = 1
        else:
            next_val = current_sum + 1
        
        seq.append(next_val)
        current_sum += next_val
    
    # The sequence is [1, 2, 4, ..., 2048]
    # Max element is 2048.
    
    # Verify validity (should always be true for superincreasing sequences)
    if not _validate(seq):
        # Fallback to explicit powers of 2 if validation fails for any reason
        seq = [1 << i for i in range(n)]
        
    return seq
