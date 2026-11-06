"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Since the objective is to maximize the number of ones,
    the optimal solution is all ones.
    
    This is a deterministic direct construction of the known global optimum.
    No search or optimization is needed; the answer is trivially "1" * N.
    
    Includes a time guard as per directive, though execution is O(1) and
    completes in microseconds, far below any timeout.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5-second safety margin before 150s kill
    
    # The optimal solution for OneMax is all ones
    result = "1" * N
    
    # Verify we haven't exceeded deadline (should never happen for this simple construction)
    if time.time() > deadline:
        # Fallback: return what we have (which is already optimal)
        return result
    
    return result
# EVOLVE-BLOCK-END
