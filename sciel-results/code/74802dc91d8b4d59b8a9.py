"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Models the construction as a deterministic bitwise OR operation between a 
    zero-initialized integer and a mask of 2^100 - 1.
    
    1. Initialize an integer to 0.
    2. Create a mask with N bits set to 1: (1 << N) - 1.
    3. Perform bitwise OR: 0 | mask.
    4. Convert to binary string using bin() and strip '0b'.
    """
    # Zero-initialized integer
    state = 0
    
    # Mask of 2^N - 1 (all N bits set to 1)
    mask = (1 << N) - 1
    
    # Deterministic bitwise OR operation
    result_int = state | mask
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(result_int)[2:]
    
    # Ensure the string is exactly N characters long
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
        
    return binary_string
# EVOLVE-BLOCK-END
