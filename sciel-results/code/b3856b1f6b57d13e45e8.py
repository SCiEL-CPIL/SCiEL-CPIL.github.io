import time

N = 100
# Explicit deadline guard as per directive. 
# The computation is O(1), so this will never trigger, 
# but it satisfies the requirement to track elapsed time.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # Safety margin below 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is maximized by setting all bits to 1.
    This function constructs the all-ones state using integer bitwise operations
    and converts it to a binary string.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return '1' * N

    # Generate an integer with the lowest 100 bits set to 1
    # (1 << N) creates a 1 followed by N zeros.
    # Subtracting 1 yields a number with N ones in its binary representation.
    mask = (1 << N) - 1
    
    # Model construction as deterministic bitwise OR with zero-initialized integer
    constructed_val = 0 | mask
    
    # Convert to binary string and strip the '0b' prefix
    result = bin(constructed_val)[2:]
    
    # Ensure the output is exactly N bits long (defensive programming)
    # For N=100, bin(2**100 - 1) is exactly 100 '1's.
    if len(result) < N:
        result = result.zfill(N)
        
    return result
# EVOLVE-BLOCK-END
