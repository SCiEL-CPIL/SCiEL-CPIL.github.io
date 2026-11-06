"""
Constructs a set of n=12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

The theoretical minimum for the largest element in a set of n integers with 
distinct subset sums is 2^(n-1). This is achieved by the powers of 2.
For n=12, the optimal set is [1, 2, 4, ..., 2048].
"""
import time

def construct():
    # Deadline guard as per directive, though computation is instantaneous.
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit
    
    # The optimal set is the powers of 2.
    # This minimizes the largest element to 2^(n-1).
    # For n=12, the largest element is 2^11 = 2048.
    n = 12
    optimal_set = [1 << i for i in range(n)]
    
    # Verify validity quickly (optional, but good for robustness if n changes)
    # For n=12, this is always true for powers of 2.
    # We skip the full 4096 subset sum check for speed, as it's mathematically guaranteed.
    
    if time.time() > deadline:
        # Should never happen, but return best known if it does
        return optimal_set
        
    return optimal_set
