"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones using NumPy vectorization."""
    # Create a NumPy array of uint8 filled with 1s
    arr = np.ones(N, dtype=np.uint8)
    # Convert each element to its string representation and join
    # Using np.char.mod for vectorized string formatting
    result = np.char.mod('1', arr)
    # Join the array of strings into a single string
    return ''.join(result)
# EVOLVE-BLOCK-END
