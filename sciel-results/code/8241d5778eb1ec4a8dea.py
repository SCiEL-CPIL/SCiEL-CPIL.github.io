"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution for OneMax is all ones. We return it directly.
    Includes an explicit deadline guard as per directive.
    """
    start_time = time.perf_counter()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones
    # Constructing the string is O(N) and instantaneous for N=100
    result = "1" * N
    
    # Explicit deadline guard
    if time.perf_counter() > deadline:
        # Return the best result found (which is optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
