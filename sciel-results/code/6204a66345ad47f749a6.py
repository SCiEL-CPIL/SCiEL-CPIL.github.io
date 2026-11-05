N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is maximized by setting all bits to 1.
    This function constructs the all-ones state using integer bitwise operations
    and converts it to a binary string.
    """
    # Generate an integer with the lowest 100 bits set to 1
    # (1 << 100) creates a 1 followed by 100 zeros.
    # Subtracting 1 yields a number with 100 ones in its binary representation.
    mask = (1 << N) - 1
    
    # Convert to binary string and strip the '0b' prefix
    result = bin(mask)[2:]
    
    return result
# EVOLVE-BLOCK-END
