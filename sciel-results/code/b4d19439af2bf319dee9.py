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
    Uses bitwise OR with a full mask to generate the all-ones state.
    """
    start_time = time.time()
    
    # Check deadline with safety margin
    if time.time() - start_time > DEADLINE_SECONDS - 10.0:
        # Fallback to a simple string if somehow delayed (unlikely)
        return '1' * N

    # Generate a mask of 2^100 - 1 (100 ones in binary)
    # This leverages Python's arbitrary precision integers
    mask = (1 << N) - 1
    
    # Convert to binary string and remove the '0b' prefix
    # bin(mask) returns a string like '0b111...1'
    result = bin(mask)[2:]
    
    # Ensure length is exactly N (bin() might not pad, but for all ones it's exact)
    # If N was not a power of 2 or similar, we might need padding, but here it's exact.
    # However, to be safe against any edge cases with bin() representation:
    if len(result) < N:
        result = result.zfill(N)
        
    return result
# EVOLVE-BLOCK-END
