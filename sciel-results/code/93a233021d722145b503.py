import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    Uses a NumPy boolean array initialized to True, then converts to string.
    """
    start_time = time.time()
    
    # Strategy: Model as NumPy boolean array initialized to True
    bits = np.ones(N, dtype=bool)
    
    # Convert boolean array to string of '1's
    # np.char is not directly used for this, but we can join or use astype
    # A boolean array of True values can be mapped to '1'
    # Using join is efficient for small N
    result = ''.join('1' if b else '0' for b in bits)
    
    # Explicit deadline guard
    if time.time() - start_time > 119:
        return result
        
    return result
# EVOLVE-BLOCK-END
