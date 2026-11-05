"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    Uses bitwise OR with an all-ones mask to derive the optimal state,
    then converts to binary string. Includes explicit deadline guard.
    """
    start = time.monotonic()
    deadline = start + 140.0  # safety margin below 150s kill
    
    # The global optimum for OneMax is all bits set to 1.
    # Model as deterministic bitwise OR between zero-initialized int and mask (2^N - 1).
    # This leverages Python's arbitrary-precision integer arithmetic.
    mask = (1 << N) - 1
    optimal_int = 0 | mask  # 0 OR (2^N - 1) = 2^N - 1
    
    # Convert to binary string, stripping '0b' prefix
    # Ensure length is exactly N (bin() of 2^N-1 gives N ones)
    result = bin(optimal_int)[2:]
    
    # Verify length is correct; if not (shouldn't happen), pad or truncate
    if len(result) != N:
        if len(result) < N:
            result = result.zfill(N)
        else:
            result = result[-N:]
    
    # Deadline guard: if we somehow spent too much time, return best found
    if time.monotonic() > deadline:
        # Fallback to direct construction
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
