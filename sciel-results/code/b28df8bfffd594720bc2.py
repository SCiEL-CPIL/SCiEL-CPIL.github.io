import time

N = 100
DEADLINE = 140.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    This implementation models the task as a set-theoretic union operation.
    It takes the union of a set of 100 unique indices with itself, 
    and maps each index to the character '1'.
    """
    start_time = time.time()
    
    # Check deadline (trivial for this O(N) op, but required by directive)
    if time.time() > DEADLINE:
        return '1' * N

    # Create a set of 100 unique integer indices
    indices = set(range(N))
    
    # Perform set union with itself (identity operation, yields same set)
    union_set = indices.union(indices)
    
    # Map each index in the sorted union set to '1' to construct the string.
    # Sorting ensures deterministic order, though order doesn't affect OneMax score.
    # Using a list comprehension for efficiency.
    result = ''.join('1' for _ in sorted(union_set))
    
    return result
# EVOLVE-BLOCK-END
