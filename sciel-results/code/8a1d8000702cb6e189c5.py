"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a 100-bit binary string with all bits set to '1'.
    This maximizes the number of ones, achieving the theoretical maximum score.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, though solution is O(1)
    
    # The optimal solution for OneMax is a string of all '1's
    best_candidate = "1" * N
    
    # Check deadline (not strictly needed for O(1) but satisfies directive pattern)
    if time.time() > deadline:
        return best_candidate
        
    return best_candidate
# EVOLVE-BLOCK-END
