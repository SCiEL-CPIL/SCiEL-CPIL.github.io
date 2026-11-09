"""Program for OneMax direct binary construction (100 bits)."""

import time

N = 100
DEADLINE = 120.0  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    
    This implementation uses a bytearray initialized with the ASCII code for '1' (0x31)
    to create a mutable sequence of 100 bytes, then decodes it to a UTF-8 string.
    This leverages low-level memory buffer operations distinct from string repetition
    or integer formatting.
    """
    # Check deadline (safety margin)
    if time.time() - START_TIME > DEADLINE - 1.0:
        # Fallback to simple string multiplication if time is critical
        return "1" * N
    
    # Create a bytearray of length N, initialized with the ASCII code for '1' (0x31)
    # bytearray([0x31] * N) creates a mutable sequence of 100 bytes, each being 0x31
    ba = bytearray([0x31] * N)
    
    # Decode the bytearray to a UTF-8 string
    # Since all bytes are 0x31 (ASCII '1'), this produces a string of 100 '1' characters
    return ba.decode('utf-8')
# EVOLVE-BLOCK-END
