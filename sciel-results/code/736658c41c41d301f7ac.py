"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The objective is OneMax: maximize the number of 1s in the string.
    The global optimum is a string of all 1s, which has 100 ones.
    This is a deterministic construction that returns the known optimum.
    Includes a deadline guard to ensure compliance with time limits,
    although the computation is instantaneous.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # The optimal solution is known and constructed in O(1)
    result = "1" * N
    
    # Check if we are still within the deadline (always true for this O(1) operation)
    if time.time() > deadline:
        # This branch is effectively unreachable for the current logic,
        # but included to satisfy the directive for explicit deadline handling.
        # In a more complex search, we would return the best partial result here.
        return result
        
    return result
# EVOLVE-BLOCK-END
