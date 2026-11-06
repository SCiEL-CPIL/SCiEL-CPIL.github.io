"""Program for OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    # Convert the boolean array to a string of '1's.
    # np.char.mod can be used, but a simpler approach is to join a list of '1's.
    # However, to leverage the array-based construction as per the directive,
    # we can convert the array to a string representation.
    # A vectorized way: convert bool array to int array, then to string.
    int_bits = bits.astype(int)
    # Convert each integer to a string and join
    return ''.join(str(b) for b in int_bits)
# EVOLVE-BLOCK-END
