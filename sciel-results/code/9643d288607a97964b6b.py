"""
Optimized program for OneMax task using direct integer bit-manipulation.
Constructs the all-ones binary string via integer formatting.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    Uses integer bitmask representation for construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Structural Change: Integer Bit-Construction
    # Represent the solution as an integer with N bits set to 1.
    # (1 << N) - 1 creates an integer with the lowest N bits set to 1.
    # Format this integer as a binary string of width N.
    
    if time.time() > deadline:
        # Fallback to simple string multiplication if time is critically low
        # (though this is effectively instantaneous)
        return '1' * N
        
    mask = (1 << N) - 1
    result = format(mask, f'0{N}b')
    
    return result
# EVOLVE-BLOCK-END
