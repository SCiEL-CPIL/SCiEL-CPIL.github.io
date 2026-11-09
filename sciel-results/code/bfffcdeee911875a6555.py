"""Program for OneMax direct binary construction (100 bits)."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The theoretical maximum is N (100) ones.
    
    This function directly constructs the global optimum, which is a string of 
    '1's of length N. This approach is optimal as it requires no search, 
    no iteration, and guarantees the maximum possible score instantly.
    
    Implementation uses bytearray initialization for low-level memory efficiency.
    """
    # Initialize a mutable byte array with the ASCII code for '1' (0x31)
    # Then decode to a UTF-8 string
    b = bytearray([0x31] * N)
    return b.decode('utf-8')
# EVOLVE-BLOCK-END
