"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective function is f(x) = sum(x_i) for x in {0, 1}^N.
    The global maximum is N, achieved uniquely by the vector of all ones.
    This function returns the optimal solution directly in O(N) time (string creation).
    """
    # Explicit deadline guard as per directive, though computation is instantaneous.
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Direct construction of the optimal solution: all bits set to 1.
    # This yields a score of 100.
    result = "1" * N
    
    # The operation is O(N) with N=100, which takes negligible time (< 1 microsecond).
    # No further search or polish is possible as this is the global optimum.
    return result
# EVOLVE-BLOCK-END
