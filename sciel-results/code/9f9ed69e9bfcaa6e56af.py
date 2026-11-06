"""
Constructs a 100-bit binary string maximizing the number of ones.
The optimal solution is a string of all ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N consisting entirely of '1's.
    Uses integer bit manipulation to generate the pattern.
    """
    # Create a 128-bit integer with all bits set to 1
    # (1 << 128) - 1 results in 2^128 - 1, which is 128 ones in binary
    mask = (1 << 128) - 1
    
    # Convert to binary string. bin() returns '0b...' prefix.
    # We take the last 100 characters to get the rightmost 100 bits.
    # Since all bits are 1, this will be '1' * 100.
    binary_str = bin(mask)[-N:]
    
    return binary_str
# EVOLVE-BLOCK-END
