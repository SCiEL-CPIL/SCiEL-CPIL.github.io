"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a 100-bit binary string with all bits set to '1'.
    This maximizes the number of ones, achieving the theoretical maximum score.
    """
    # Explicit deadline guard as per directive
    start_time = time.time()
    deadline = start_time + 10.0
    
    # The optimal solution for OneMax is a string of all '1's
    best_candidate = "1" * N
    
    # Return immediately as solution is O(1) and well within deadline
    return best_candidate
# EVOLVE-BLOCK-END
