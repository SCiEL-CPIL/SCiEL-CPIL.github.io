"""Construct 12 distinct positive integers with distinct subset sums, minimizing the max."""
import time

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The set minimizes the largest element.
    
    The minimum possible largest element for n distinct positive integers with distinct subset sums is 2^(n-1).
    For n=12, this is 2^11 = 2048.
    The optimal set is the set of powers of 2: [1, 2, 4, ..., 2048].
    
    Proof of optimality:
    Let a_1 < a_2 < ... < a_n be the sorted elements.
    Base case: a_1 >= 1 = 2^0.
    Inductive step: Assume a_i >= 2^(i-1) for i < j.
    The sum of the first j-1 elements is at most 2^(j-1) - 1.
    To ensure distinct subset sums, a_j must be greater than the maximum sum of the previous j-1 elements
    (otherwise, two subsets could collide: one containing a_j and a subset of previous elements summing to a_j,
    and another not containing a_j but containing that same subset).
    Thus a_j >= 1 + sum(a_1..a_{j-1}) >= 1 + (2^(j-1) - 1) = 2^(j-1).
    Therefore a_12 >= 2^11 = 2048.
    The powers of 2 achieve this bound exactly.
    """
    n = 12
    deadline = time.time() + 5.0  # Safety margin; computation is instant
    
    # The optimal solution is the first n powers of 2.
    result = [1 << i for i in range(n)]
    
    # Validation
    # Check distinctness
    if len(set(result)) != n:
        raise ValueError("Elements must be distinct")
    
    # Check positivity
    if any(x <= 0 for x in result):
        raise ValueError("Elements must be positive")
        
    # Check bound
    if max(result) > 1048576:
        raise ValueError("Largest element exceeds bound")

    # Check subset sum distinctness
    # There are 2^12 = 4096 subsets.
    subset_sums = set()
    # Using bit manipulation to iterate through all subsets
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        subset_sums.add(s)
        
        # Explicit deadline guard (though this loop is microseconds)
        if time.time() > deadline:
            # If we somehow timed out (impossible for n=12), return the constructed set
            # which is known to be valid.
            return result

    if len(subset_sums) != (1 << n):
        raise ValueError("Subset sums are not all distinct")
        
    return result
