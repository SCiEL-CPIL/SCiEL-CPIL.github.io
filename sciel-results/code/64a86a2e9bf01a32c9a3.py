"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Directly constructs the optimal 100-bit binary string for OneMax.
    The optimal solution is a string of length N consisting entirely of '1's.
    This is the global optimum, so no search or evolution is required.
    
    Strategy: Use bitwise OR with a mask (2^N - 1) to generate the integer
    representation of all ones, then convert to binary string.
    """
    # Deadline guard (though this operation is instantaneous)
    if time.time() - START_TIME > TIME_LIMIT - 1.0:
        return "1" * N

    # Generate mask for N bits: 2^N - 1
    # For N=100, this is a large integer, but Python handles it easily.
    mask = (1 << N) - 1
    
    # Bitwise OR with 0 is just the mask itself, but we follow the directive's phrasing
    # "deterministic bitwise OR operation between a zero-initialized integer and a mask"
    state = 0 | mask
    
    # Convert to binary string, removing the '0b' prefix
    binary_str = bin(state)[2:]
    
    # Ensure length is N (padding if necessary, though 2^N-1 has exactly N ones)
    if len(binary_str) < N:
        binary_str = binary_str.zfill(N)
        
    return binary_str
# EVOLVE-BLOCK-END
