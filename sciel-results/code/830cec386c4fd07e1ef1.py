"""Program for OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    # Use np.where to vectorize the conversion of boolean values to string characters.
    # This maps True to '1' and False to '0'.
    char_bits = np.where(bits, '1', '0')
    # Join the array of single-character strings into a single binary string.
    return ''.join(char_bits)
# EVOLVE-BLOCK-END
