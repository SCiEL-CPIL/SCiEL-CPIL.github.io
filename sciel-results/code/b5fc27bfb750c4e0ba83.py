import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the optimal 100-bit binary string using NumPy vectorized allocation.
    
    The objective is to maximize the number of '1's in a binary string of length N.
    The maximum possible number of '1's is N, achieved by the string consisting of
    all '1's. This is the unique global optimum for the OneMax problem.
    
    This function leverages NumPy's efficient memory allocation to create a boolean
    array of True values (representing '1's) and converts it to the required string
    format. A deadline guard is included per the compute budget directive.
    """
    # Track elapsed time for deadline guard (safety margin)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Structural change: Use NumPy boolean array for vectorized construction
    # np.ones(N, dtype=bool) creates an array of N True values efficiently
    bool_array = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's and '0's
    # astype('U1') converts True->'1', False->'0'
    result = bool_array.astype('U1').join('')
    
    # Verify we're still within budget (this will always pass for this problem)
    if time.time() > deadline:
        # Fallback: return best available (which is already the optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
