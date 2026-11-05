"""
Optimal OneMax solution: Construct the string of all 1s.
"""

import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150  # Seconds

def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string consisting entirely of '1's.
    This function directly returns that string.
    Includes a deadline guard as per directive, although computation is trivial.
    """
    if time.time() - START_TIME > TIME_LIMIT - 1:
        # Fallback to a valid result if time limit is breached (unlikely for this task)
        return "1" * N
        
    return "1" * N
