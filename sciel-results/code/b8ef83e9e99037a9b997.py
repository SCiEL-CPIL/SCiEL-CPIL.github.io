"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    The optimal solution for OneMax is a string of all ones.
    This function directly constructs that string.
    """
    start_time = time.time()
    
    # The global optimum for OneMax is a string of N ones.
    # Direct construction is the most efficient method.
    result = '1' * N
    
    # Explicit deadline guard
    # Although this operation is instantaneous, we maintain the guard
    # to satisfy the requirement for a safety margin before the limit.
    if time.time() - start_time > DEADLINE_SECONDS - 10.0:
        return result
        
    return result
# EVOLVE-BLOCK-END
