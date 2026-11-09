"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones. We return it directly.
    Includes a deadline guard as required.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The OneMax problem with 100 bits has a clear global optimum:
    # all bits set to 1, giving a score of 100.
    # No search is needed; we can construct the optimal solution directly.
    result = "1" * N
    
    # Verify we're within deadline (should always be true for this O(1) solution)
    if time.time() < deadline:
        return result
    else:
        # Fallback: return what we have (which is already optimal)
        return result
# EVOLVE-BLOCK-END
