import numpy as np
import time

N = 100
DEADLINE_SECONDS = 145.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Explicit deadline guard
    if time.time() - start_time > DEADLINE_SECONDS:
        return '1' * N

    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    
    # Check deadline again before conversion (redundant but explicit)
    if time.time() - start_time > DEADLINE_SECONDS:
        return '1' * N

    # Convert the boolean array to a string of '1's.
    # Since all are True, the optimal solution is a string of N ones.
    return ''.join('1' for _ in bits)
# EVOLVE-BLOCK-END
