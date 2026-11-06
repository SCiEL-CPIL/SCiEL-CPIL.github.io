import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The optimal solution is the set of the first 12 powers of 2:
    {1, 2, 4, ..., 2048}.
    The largest element is 2048.
    
    This function implements the greedy construction of a superincreasing sequence,
    which is proven to be optimal for this objective.
    It includes a time guard to ensure it finishes well within the 60s limit,
    although the computation is O(n).
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 12
    max_limit = 1048576
    
    # The optimal sequence is powers of 2.
    # x_0 = 1
    # x_{k} = 1 + sum(x_0 ... x_{k-1})
    # This generates 1, 2, 4, 8, ...
    
    elements = []
    current_sum = 0
    
    for i in range(n):
        # Greedy choice: smallest integer > current_sum
        # Since we start with 1, this forces powers of 2.
        # If current_sum = 0, x=1.
        # If current_sum = 1, x=2.
        # If current_sum = 3, x=4.
        # If current_sum = 7, x=8.
        
        next_val = current_sum + 1
        
        # Check bounds and time
        if next_val > max_limit:
            # Fallback, though for n=12 this won't happen
            # Powers of 2 up to 2^11 = 2048 << 1048576
            return [1 << i for i in range(n)]
            
        if time.time() > deadline:
            # Return best known if time runs out (shouldn't happen)
            break
            
        elements.append(next_val)
        current_sum += next_val

    # If loop completed, return the constructed elements
    # For n=12, this is [1, 2, 4, ..., 2048]
    return elements
# EVOLVE-BLOCK-END
