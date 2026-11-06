"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The optimal solution for OneMax is the string of all ones.
    # Direct construction yields the maximum possible score immediately.
    result = "1" * N
    
    # Check if we are still within the deadline (always true for this O(N) operation)
    if time.time() < deadline:
        return result
    else:
        # Fallback (should never happen for N=100)
        return "1" * N
# EVOLVE-BLOCK-END
