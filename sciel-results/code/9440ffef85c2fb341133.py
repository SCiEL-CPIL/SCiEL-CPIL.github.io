import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Create a NumPy boolean array of size N, initialized to True (1)
    arr = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's and '0's
    # astype('U1') converts True to '1' and False to '0'
    binary_chars = arr.astype('U1')
    
    # Join the array of characters into a single string
    return "".join(binary_chars)
# EVOLVE-BLOCK-END
