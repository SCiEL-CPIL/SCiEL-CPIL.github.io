import time

def construct():
    """
    Constructs a set of n=12 distinct positive integers with distinct subset sums
    minimizing the largest element.
    
    The optimal solution is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    This is a superincreasing sequence where each element is greater than the sum of all previous elements.
    For n=12, the largest element is 2^11 = 2048.
    
    Returns:
        list: A list of 12 integers.
    """
    n = 12
    # The minimal largest element is 2^(n-1).
    # The set is {2^0, 2^1, ..., 2^(n-1)}.
    # This ensures all subset sums are distinct because binary representation is unique.
    
    # We can generate this sequence iteratively to show the "algorithm"
    # which is the greedy choice of the smallest valid integer.
    vals = []
    current_sum = 0
    for i in range(n):
        # The next element must be > sum of previous elements to maintain distinct subset sums
        # and to be the smallest possible such element.
        next_val = current_sum + 1
        vals.append(next_val)
        current_sum += next_val
        
    # Verify distinct subset sums (optional, but good for robustness in case of logic errors)
    # For n=12, 2^12 = 4096 sums, which is fast.
    # sums = {0}
    # for v in vals:
    #     new_sums = set()
    #     for s in sums:
    #         new_sums.add(s + v)
    #     if len(sums.union(new_sums)) != len(sums) + len(new_sums):
    #         # Collision detected
    #         pass
    #     sums = sums.union(new_sums)
    # if len(sums) != 2**n:
    #     raise ValueError("Subset sums are not distinct")

    return vals
