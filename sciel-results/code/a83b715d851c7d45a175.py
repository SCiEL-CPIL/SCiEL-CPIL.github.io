"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120
    
    # Create a NumPy boolean array of length N, initialized to True (1)
    # This leverages vectorized memory allocation.
    arr = np.ones(N, dtype=bool)
    
    # Convert to binary string: True -> '1', False -> '0'
    # Using np.where to map booleans to characters, then joining
    # np.char.join is efficient for this.
    result = np.char.join('', np.where(arr, '1', '0'))
    
    # Safety margin check
    if time.time() > deadline - 5:
        return result
        
    return result
# EVOLVE-BLOCK-END
