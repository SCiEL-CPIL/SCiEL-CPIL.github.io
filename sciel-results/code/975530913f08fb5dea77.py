"""Direct OneMax construction: return the optimal 100-bit all-ones string."""

import time
import numpy as np

N = 100
TIME_LIMIT = 120.0  # Seconds
SAFETY_MARGIN = 1.0 # Seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Return a 100-bit binary string with the maximum possible number of ones.
    
    Uses a NumPy boolean array initialized to True to represent the optimal
    genotype, then converts to a string of '1's. This leverages vectorized
    memory representation as a distinct computational path to the optimal solution.
    """
    start_time = time.time()
    
    # Deadline guard: check if we are already past the safety margin
    if time.time() - start_time > (TIME_LIMIT - SAFETY_MARGIN):
        # Fallback: return all ones (still optimal) if somehow out of time
        return "1" * N

    # Construct the optimal solution using NumPy boolean array
    # A boolean array of True values represents all-ones bits
    bits = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's
    # np.chararray or manual conversion
    # Using join for clarity and speed
    result = ''.join('1' if b else '0' for b in bits)
    
    # Ensure we return before the hard limit
    if time.time() - start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return result
        
    return result
# EVOLVE-BLOCK-END
