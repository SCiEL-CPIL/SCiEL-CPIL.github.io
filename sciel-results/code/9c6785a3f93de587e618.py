"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE = 150.0  # Seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    
    Includes a deadline guard as per directive, though the operation is O(1) effectively.
    """
    start_time = time.time()
    
    # The optimal solution for OneMax is a string of all ones.
    # We construct it directly.
    result = "1" * N
    
    # Check deadline (will never be exceeded for N=100)
    if time.time() - start_time > DEADLINE:
        # In a real search, we would return the best found so far.
        # Here, we just return the result as it is already optimal.
        pass
        
    return result
# EVOLVE-BLOCK-END
