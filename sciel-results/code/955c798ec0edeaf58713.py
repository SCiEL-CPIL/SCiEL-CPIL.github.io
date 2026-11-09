"""Program for OneMax direct binary construction (100 bits)."""
import time

N = 100
START_TIME = time.monotonic()
DEADLINE = 115.0  # safety margin before 120s limit
# Pre-compute the optimal solution to minimize function call overhead
OPTIMAL_RESULT = '1' * N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    OneMax objective: maximize count of '1' bits in a length-N string.
    The unique global optimum is all ones. We return the pre-computed
    optimal string directly. An explicit deadline guard ensures we
    return a valid result even under pathological timing conditions.
    """
    # Explicit deadline guard
    if (time.monotonic() - START_TIME) > DEADLINE:
        return OPTIMAL_RESULT
        
    return OPTIMAL_RESULT
# EVOLVE-BLOCK-END
