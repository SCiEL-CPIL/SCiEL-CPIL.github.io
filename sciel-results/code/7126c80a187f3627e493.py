import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
# Pre-allocate the solution at module load time
_SOLUTION_ARRAY = np.ones(N, dtype=np.int8)
_SOLUTION_STRING = None

def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    """
    global _SOLUTION_STRING
    
    # Safety guard
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    if _SOLUTION_STRING is None:
        # Convert the pre-allocated array to a string.
        # Using a list comprehension on the numpy array is fast for small N.
        _SOLUTION_STRING = "".join(map(str, _SOLUTION_ARRAY))
        
    elapsed = time.time() - start_time
    if elapsed + safety_margin > deadline:
        return _SOLUTION_STRING

    return _SOLUTION_STRING
# EVOLVE-BLOCK-END
