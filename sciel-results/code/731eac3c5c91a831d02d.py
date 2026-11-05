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
    
    The global optimum for OneMax is the all-ones string. 
    This implementation uses direct string repetition, which is the most 
    efficient and robust method for this trivial construction.
    
    Includes an explicit deadline guard to ensure compliance with time limits,
    although the operation is constant time.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # The optimal solution for OneMax is a string of all ones.
    # This is a direct construction based on the problem definition.
    # Using string repetition is simpler and faster than integer bit manipulation
    # followed by bin() conversion and prefix stripping.
    result = "1" * N
    
    # Explicit deadline guard as per directive
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
