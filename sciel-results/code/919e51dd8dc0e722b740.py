"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Since the objective is OneMax (maximize number of 1s) and the only constraint
    is that the output is a binary string of length N=100, the optimal solution
    is simply a string of all 1s.
    
    Uses Python's arbitrary-precision integer arithmetic:
    - Create a mask of 2^N - 1 (all N bits set to 1)
    - Convert to binary string using bin()
    - Strip the '0b' prefix
    - Pad with leading zeros if necessary (though not needed for all-ones)
    
    This produces a string of N '1's in constant time.
    """
    # Create an integer with all N bits set to 1
    mask = (1 << N) - 1
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(mask)[2:]
    
    # Ensure the string is exactly N characters long
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
    
    return binary_string
# EVOLVE-BLOCK-END
