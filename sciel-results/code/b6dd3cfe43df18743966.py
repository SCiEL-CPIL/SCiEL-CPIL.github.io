"""
OneMax direct binary construction (100 bits)

Constructs a 100-bit binary string maximizing ones.
The optimal solution for OneMax is a string of all '1's.
"""
import time
import numpy as np

N = 100
# Deadline guard: 120s limit, safety margin 10s -> 110s
DEADLINE_SECONDS = 110.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy:
    1. Check elapsed time against the deadline guard.
    2. Return the optimal string of '1's repeated N times.
    
    This is the global optimum for the OneMax problem.
    """
    start_time = time.time()
    
    # The optimal solution for OneMax is all ones.
    # We can construct this directly.
    # Using string multiplication is faster and simpler than NumPy char operations.
    best_result = '1' * N
    
    # Deadline guard: 
    # Since this operation is O(N) and N=100, it completes in microseconds.
    # The guard is included to satisfy the directive's requirement for an explicit deadline check,
    # ensuring that if this were a more complex search, it would return the best result found.
    if time.time() - start_time > DEADLINE_SECONDS:
        # In a real search scenario, we would return the best candidate found so far.
        # Here, we have the global optimum, so we return it.
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
