import time

N = 100
DEADLINE_SECONDS = 145.0

def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    Direct construction: return a string of N '1's.
    Includes an explicit deadline guard.
    """
    t0 = time.monotonic()
    
    # Optimal solution for OneMax is all ones
    result = '1' * N
    
    # Deadline guard: if we've exceeded our safety margin, return what we have
    if time.monotonic() - t0 > DEADLINE_SECONDS:
        return result
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
