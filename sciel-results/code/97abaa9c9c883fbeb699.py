"""OneMax direct binary construction achieving global optimum."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string that maximizes the number of ones.
    
    Uses iterative character accumulation with an explicit deadline guard.
    The global optimum for OneMax is the string of N consecutive '1' characters.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    t0 = time.time()
    deadline = t0 + 118.0  # 2-second safety margin before 120s wall-clock limit

    # Direct construction of the optimal string
    # This is the global optimum for OneMax
    result = '1' * N

    # Deadline guard: if we somehow exceed time (unlikely for this operation),
    # return the result which is already optimal.
    if time.time() - t0 > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
