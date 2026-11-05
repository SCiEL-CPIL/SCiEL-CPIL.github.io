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
    This is the global optimum for the OneMax objective.
    
    Uses a deterministic bitwise OR operation between a zero-initialized integer
    and a mask of 2^N - 1. The resulting integer is converted to a binary string.
    This leverages Python's arbitrary-precision integer arithmetic.
    
    Includes explicit deadline guard for safety.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Construction via integer mask
    # mask = 2^N - 1 creates an integer with N bits set to 1
    mask = (1 << N) - 1
    
    # Convert to binary string, removing the '0b' prefix
    # zfill ensures the length is exactly N (though 2^N-1 has exactly N bits for N>0)
    result = bin(mask)[2:].zfill(N)
    
    # Explicit deadline guard
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
