"""
Task: OneMax direct binary construction (100 bits)
Objective: Maximize the number of ones in a 100-bit binary string.
Solution: Return a string of 100 ones.
"""

N = 100

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    This maximizes the OneMax objective function.
    Includes a deadline guard as per directive, though computation is instantaneous.
    """
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit
    
    # The optimal solution for OneMax(N=100) is a string of 100 '1's.
    # This is computed in O(N) time, which is negligible.
    result = "1" * N
    
    # Explicit deadline check (will always pass immediately for this static solution)
    if time.time() > deadline:
        # Fallback to best known result (which is the optimal result anyway)
        return result
        
    return result
# EVOLVE-BLOCK-END
