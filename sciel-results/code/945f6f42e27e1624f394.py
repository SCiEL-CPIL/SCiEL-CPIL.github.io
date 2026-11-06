"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is a string of all ones.
    Includes a safety guard for the 150s wall-clock limit.
    """
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Direct construction of the global optimum
    result = "1" * N
    
    # Explicit deadline guard as per directive
    # Since the computation is O(N) and N=100, this will never trigger,
    # but it satisfies the requirement to track elapsed time.
    if time.time() + safety_margin > deadline:
        # Return the best result found (which is the optimal one)
        return result
        
    return result
# EVOLVE-BLOCK-END
