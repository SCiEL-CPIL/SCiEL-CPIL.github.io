"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones using NumPy vectorization."""
    # Create a NumPy array of uint8 filled with 1s
    arr = np.ones(N, dtype=np.uint8)
    # Cast the array to a NumPy array of Unicode strings (length 1)
    # This is a direct cast, avoiding per-element formatting overhead of np.char.mod
    str_arr = arr.astype('U1')
    # Join the array of strings into a single string
    return ''.join(str_arr)
# EVOLVE-BLOCK-END
