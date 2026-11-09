"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is to maximize the count of '1's in a binary string.
    For a fixed length N, the global optimum is a string consisting entirely of '1's.
    This solution directly returns the optimal string.
    """
    # Deadline guard for safety, though this operation is instantaneous
    start_time = time.time()
    deadline = start_time + 119.0  # 119s margin within 120s limit
    
    # The optimal solution for OneMax is all ones
    result = "1" * N
    
    # Check deadline (will never trigger for N=100)
    if time.time() > deadline:
        # In case of unexpected delay, return the best found (which is the optimal)
        return result
        
    return result
# EVOLVE-BLOCK-END
