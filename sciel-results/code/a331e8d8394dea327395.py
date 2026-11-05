"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    Uses integer bitwise OR to generate the all-ones state, then converts to binary string.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Structural change: Use integer arithmetic instead of string repetition.
    # Initialize to zero
    x = 0
    # Create a mask with N bits set to 1: 2^N - 1
    mask = (1 << N) - 1
    # Bitwise OR to set all bits
    x = x | mask
    
    # Convert integer to binary string, stripping the '0b' prefix
    result = bin(x)[2:]
    
    # Ensure the length is exactly N (pad with zeros if necessary, though not needed here)
    if len(result) < N:
        result = result.zfill(N)
    elif len(result) > N:
        result = result[-N:]
        
    # Explicit deadline guard as per directive
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
