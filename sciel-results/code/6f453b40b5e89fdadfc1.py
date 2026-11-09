"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    Since the objective is deterministic and known (maximize count of '1's),
    we directly return the global optimum.
    """
    # Deadline guard: track elapsed time and return best result with safety margin
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit
    
    # Direct construction of the known global optimum (all ones)
    # This eliminates any search overhead and guarantees the maximum possible score of 100
    result = "1" * N
    
    # Verify length constraint
    if len(result) != N:
        # Fallback: ensure correct length (should never happen)
        result = "1" * N
    
    # Check deadline (should never be triggered for this simple construction)
    if time.time() > deadline:
        # Return the best result found (which is the optimal solution)
        return result
    
    return result
# EVOLVE-BLOCK-END
