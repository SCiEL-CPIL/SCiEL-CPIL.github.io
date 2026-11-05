"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Uses a bitwise approach to generate the integer with all bits set,
    then converts it to a binary string representation padded to N bits.
    """
    # Create an integer with N bits set to 1
    mask = (1 << N) - 1
    # Convert to binary string, remove '0b' prefix, and pad with zeros to ensure length N
    # Since mask has exactly N bits (all 1s), zfill is technically redundant for length
    # if we just use bin(mask)[2:], but it's safe.
    # bin(mask) for 100 bits is '0b' + 100 ones.
    # So bin(mask)[2:] is exactly 100 ones.
    return bin(mask)[2:]
# EVOLVE-BLOCK-END
