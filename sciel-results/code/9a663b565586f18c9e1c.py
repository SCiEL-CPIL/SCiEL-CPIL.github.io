"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is to maximize the number of 1s in a binary string.
    The global optimum is a string of all 1s, which yields a score of N.
    This is achieved in constant time with no constraints other than length.
    """
    # Track elapsed time for safety margin
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s limit
    
    # The optimal solution for OneMax is all ones
    # Using bitwise OR with a mask to generate the all-ones state
    # 2^100 - 1 gives a number with exactly 100 bits all set to 1
    mask = (1 << N) - 1
    state = 0 | mask
    
    # Convert integer to binary string, stripping the '0b' prefix
    result = bin(state)[2:]
    
    # Ensure correct length (pad if necessary, though shouldn't be needed for 2^100-1)
    if len(result) < N:
        result = result.zfill(N)
    elif len(result) > N:
        result = result[-N:]
    
    # Verify length (sanity check, should always be true)
    assert len(result) == N
    
    # Return immediately; no search needed
    return result
# EVOLVE-BLOCK-END
