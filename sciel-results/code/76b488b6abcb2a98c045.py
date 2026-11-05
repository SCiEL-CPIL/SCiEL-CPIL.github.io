"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    Uses bitwise OR with an all-ones mask to generate the integer state,
    then converts to a binary string. Protected by an explicit deadline guard.
    """
    start_time = time.time()
    
    # The optimal bit for OneMax is always '1'.
    # We model the construction as a deterministic bitwise OR operation
    # between a zero-initialized integer and a mask of 2^N - 1.
    # This leverages Python's arbitrary-precision integer arithmetic.
    
    # Mask of N ones: 2^N - 1
    # For N=100, this is a 100-bit integer with all bits set to 1.
    mask = (1 << N) - 1
    
    # Zero-initialized integer
    state = 0
    
    # Bitwise OR to set all bits to 1
    state = state | mask
    
    # Convert the resulting integer to a binary string.
    # bin() returns a string like '0b111...1'.
    # We strip the '0b' prefix.
    # We also ensure the length is exactly N by zfill, although for all-ones
    # it should already be N. This is a safety measure for format strictness.
    result = bin(state)[2:].zfill(N)
    
    # Explicit deadline guard
    # Check if we have exceeded the safety margin.
    # If so, return the result (which is already optimal).
    # Since this operation is constant time, this branch is effectively never taken
    # unless the system is severely stalled, but it satisfies the requirement.
    if time.time() - start_time > DEADLINE_SECONDS - 10.0:
        return result
        
    return result
# EVOLVE-BLOCK-END
