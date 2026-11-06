"""
Rewritten program for OneMax (100 bits).
Uses integer bit-manipulation to generate the optimal all-ones string.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Structural Change: Use Python's native big-integer arithmetic.
    # Create an integer with 128 bits all set to 1.
    # (1 << 128) - 1 creates a mask of 128 ones.
    mask = (1 << 128) - 1
    
    # The problem requires 100 bits.
    # We can simply take the rightmost 100 bits of the 128-bit mask.
    # Alternatively, we can construct a 100-bit mask directly: (1 << 100) - 1.
    # The prompt suggests: "Model the binary string as a 128-bit integer ... then extract the rightmost 100 bits using bin(x)[-100:]"
    
    # Let's follow the specific instruction:
    # 1. Create 128-bit integer with all ones
    x = mask
    
    # 2. Extract rightmost 100 bits
    # bin(x) returns a string like '0b111...1'
    # Slicing [-100:] gets the last 100 characters.
    # Since x has 128 ones, the last 100 are all ones.
    result = bin(x)[-100:]
    
    # Safety check: Ensure length is exactly 100.
    # If for some reason the logic failed, bin(x) might be shorter, but 128 > 100 so it's safe.
    # If we used (1 << 100) - 1, bin(x) would be '0b' + 100 ones, so [-100:] would be correct too.
    
    # Check deadline (safety margin)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 1.0:
        return result
        
    return result
# EVOLVE-BLOCK-END
