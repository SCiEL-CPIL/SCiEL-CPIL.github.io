"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is all ones.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # For OneMax, the global optimum is simply all ones.
    # This is a deterministic construction that achieves the maximum possible score.
    # No search is needed; the answer is known analytically.
    
    # Create a NumPy array of all ones (integers)
    # Then convert to a string of '1's
    # This is O(N) and extremely fast.
    
    # Simplest and most robust approach:
    # Just return a string of N ones.
    result = "1" * N
    
    # Verify length (defensive check)
    if len(result) != N:
        # Fallback (should never happen)
        result = "1" * N
    
    return result
# EVOLVE-BLOCK-END
