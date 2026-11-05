"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    Includes a time deadline guard to ensure safe termination.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin below the 150s hard limit
    
    # The optimal solution for OneMax is a string of all ones.
    # This takes O(N) time to construct, which is negligible.
    best_result = "1" * N
    
    # Check time to satisfy the guard requirement, though 
    # the operation above is instantaneous.
    if time.time() - start_time > deadline:
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
