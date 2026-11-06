"""OneMax direct binary construction (100 bits).

This module implements a deterministic construction of the optimal solution
for the OneMax problem, which requires maximizing the number of ones in a
binary string of length N. The optimal solution is a string of all ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the optimal 100-bit binary string for the OneMax objective.
    
    The objective is to maximize the count of '1' bits. The theoretical maximum
    is N, achieved by a string consisting entirely of '1's. This function
    returns that string directly.
    
    A deadline guard is included to ensure compliance with compute budget
    constraints, although the direct construction is O(N) and instantaneous.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Direct construction of the global optimum
    result = '1' * N
    
    # Check deadline (will always pass for N=100)
    if time.time() > deadline:
        # Fallback should never trigger, but ensures valid return type
        return result
        
    return result
# EVOLVE-BLOCK-END
