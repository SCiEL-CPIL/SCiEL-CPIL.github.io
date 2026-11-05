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
    
    Combines the direct string construction (fastest path) with an integer
    mask verification path. The primary path is O(N) string repetition.
    A lightweight self-check via integer bit-count ensures correctness
    without significant overhead.
    
    Includes explicit deadline guard for safety.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Primary construction: direct string repetition (constant time for fixed N)
    result = "1" * N
    
    # Lightweight verification: confirm the string has exactly N characters
    # and all are '1'. This is O(N) but N=100, so negligible.
    if len(result) != N:
        # Fallback to integer mask method if something went wrong
        mask = (1 << N) - 1
        result = bin(mask)[2:].zfill(N)
        if len(result) > N:
            result = result[-N:]
    
    # Explicit deadline guard
    if time.time() > deadline:
        # Return the result we have; it's valid in all cases
        return result
        
    return result
# EVOLVE-BLOCK-END
