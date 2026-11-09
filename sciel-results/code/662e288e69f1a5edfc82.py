"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for the OneMax problem.
    The OneMax objective is to maximize the number of ones in the string.
    The global optimum is a string of all ones.
    """
    # Explicit deadline guard as per directive, though computation is O(1).
    # This ensures we return within the 120s limit (or 150s kill switch).
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # The optimal solution for OneMax is all 1s.
    # This is the global maximum.
    optimal_string = "1" * N
    
    # Check if we are still within time (always true for O(1) op, but follows directive)
    if time.time() < deadline:
        return optimal_string
    else:
        # Fallback if somehow time ran out (impossible for this op, but safe)
        return optimal_string
# EVOLVE-BLOCK-END
