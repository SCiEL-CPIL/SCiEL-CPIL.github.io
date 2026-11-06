"""
Optimized program for OneMax task.
Constructs a 100-bit binary string maximizing the number of ones.
"""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
# Pre-allocate the optimal solution at module load time
_ONES_ARRAY = np.ones(N, dtype=np.uint8)
_ONES_STRING = "1" * N

def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    The 'model' is the constant length N, and the 'optimization' 
    yields the parameter vector of all ones.
    """
    # Explicit deadline guard as per directive
    # Although this function is O(1), we maintain the guard structure
    start_time = time.time()
    deadline = 150.0 # Wall-clock limit
    safety_margin = 5.0
    
    # Return the pre-computed optimal string
    result = _ONES_STRING
    
    # Safety guard: Check if we have exceeded the deadline
    if time.time() + safety_margin > deadline:
         return result
         
    return result
# EVOLVE-BLOCK-END
