import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    Uses the analytical solution: all bits set to 1.
    Includes an explicit deadline guard as per directive.
    """
    start_time = time.time()
    # Explicit deadline guard with a safety margin (150s limit, use 140s)
    deadline = start_time + 140.0
    
    # Analytical optimum for OneMax: all ones
    # This operation is O(N) and completes in microseconds
    result = "1" * N
    
    # Check if we are still within the safe time limit
    # This check is redundant for performance but satisfies the directive
    if time.time() > deadline:
        # Fallback: return a valid string, though this should never happen
        # for such a trivial computation
        return "0" * N
        
    return result
# EVOLVE-BLOCK-END
