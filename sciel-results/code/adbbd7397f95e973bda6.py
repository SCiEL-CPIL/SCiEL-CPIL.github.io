import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string of all 1s.
    We directly construct this solution in O(N) time.
    Includes an explicit deadline guard as required by the compute budget directive.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Direct construction of the optimal solution: all bits set to 1
    # This is the global optimum, so no further search is needed or possible.
    result = "1" * N
    
    # Explicit deadline guard: check if we have exceeded the budget.
    # For this direct construction, we will never exceed it, but the check
    # satisfies the directive to maintain an explicit guard.
    if time.time() > deadline:
        # Fallback: return the constructed result even if deadline exceeded
        return result
        
    return result
# EVOLVE-BLOCK-END
