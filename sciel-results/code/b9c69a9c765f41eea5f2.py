"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Uses Python's arbitrary-precision integer arithmetic:
    - Create a mask of 2^100 - 1 (all 100 bits set to 1)
    - Convert to binary string using bin()
    - Strip the '0b' prefix
    
    This produces a string of 100 '1's in constant time.
    """
    # Create an integer with all N bits set to 1
    mask = (1 << N) - 1
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(mask)[2:]
    
    # Ensure the string is exactly N characters long (pad with leading zeros if necessary)
    # Since mask = 2^N - 1, bin(mask) will be '0b' followed by N ones, so no padding needed
    # But for safety, we can right-pad or left-pad to ensure length N
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
    
    return binary_string
# EVOLVE-BLOCK-END
