import time

# EVOLVE-BLOCK-START
def construct():
    # The optimal set of n distinct positive integers with distinct subset sums
    # is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    # For n=12, the largest element is 2^11 = 2048.
    
    n = 12
    max_limit = 1048576
    
    # Generate the optimal sequence
    result = [1 << i for i in range(n)]
    
    # Verify the largest element is within bounds
    if result[-1] > max_limit:
        # Fallback: this shouldn't happen for n=12
        pass
        
    return result
# EVOLVE-BLOCK-END
