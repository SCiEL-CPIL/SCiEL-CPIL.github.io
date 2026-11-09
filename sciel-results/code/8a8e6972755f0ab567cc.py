"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the optimal 100-bit binary string.
    
    The objective is to maximize the number of '1's in a 100-bit string.
    The global optimum is a string of 100 '1's, which yields a score of 100.
    This function returns that string directly.
    
    Implementation strategy: Represent the solution as a single 128-bit integer
    constructed via bitwise left-shifts from a base-2 power, then extract the
    lower 100 bits as a binary string using format(..., '0100b'). This leverages
    arbitrary-precision integer arithmetic as a distinct computational pathway
    from string repetition or array allocation to generate the optimal all-ones
    genotype.
    
    Deadline guard: Track elapsed time and return the best result found with a
    safety margin before the 150s limit. Since this is O(1) construction, the
    deadline is trivially met, but we include the guard for robustness.
    """
    start_time = time.time()
    
    # Safety margin: return if we're getting close to the 150s limit
    # (This is effectively never triggered for this O(1) construction)
    if time.time() - start_time > 145:
        # Fallback: simple string repetition
        return "1" * N
    
    # Construct a 128-bit integer with the lower 100 bits set to 1
    # 2^100 - 1 gives us exactly 100 ones in the lower bits
    # We can also think of it as: start with 1, left-shift 99 times, add 1, etc.
    # But the most direct "bitwise left-shifts from a base-2 power" approach:
    # Base-2 power: 2^100 = 1 << 100
    # 2^100 - 1 = all ones in lower 100 bits
    # To make it explicitly a 128-bit pathway, we can mask to 128 bits:
    # (2^100 - 1) & ((1 << 128) - 1) — this is just 2^100 - 1 since 100 < 128
    
    # Construct via left-shifts from base-2 power:
    # Start with base = 2 (2^1), left-shift 99 times to get 2^100, subtract 1
    base = 2
    for _ in range(99):
        base = base << 1  # base = 2^100 after 99 shifts
    all_ones_100 = base - 1  # 2^100 - 1, which has 100 ones
    
    # Mask to 128-bit integer (redundant but follows the strategy)
    mask_128 = (1 << 128) - 1
    all_ones_100 = all_ones_100 & mask_128
    
    # Extract lower 100 bits as binary string
    result = format(all_ones_100, '0100b')
    
    # Verify length is exactly 100
    if len(result) != N:
        # Fallback
        result = "1" * N
    
    # Final deadline check
    if time.time() - start_time > 145:
        return "1" * N
    
    return result
# EVOLVE-BLOCK-END
