"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns the global optimum: a string of 100 '1's.
    Includes an explicit deadline guard as per directive.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # The optimal solution is known and constructed in O(1)
    result = "1" * N
    
    # Explicit deadline guard
    if time.time() - start_time > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
