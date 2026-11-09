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
    # Deadline guard: although this operation is O(N) and extremely fast,
    # we include a safety check as per directive.
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds safety margin within 120s limit
    
    # Directly construct the optimal string
    best_result = "1" * N
    
    # In a real search scenario, we might loop here, but for OneMax 
    # the optimum is found immediately.
    # If this were a complex search, we would check time here.
    # time.sleep(0) # No sleep needed, return immediately.
    
    if time.time() < deadline:
        return best_result
    else:
        # Fallback (should never happen for this simple task)
        return "0" * N
# EVOLVE-BLOCK-END
