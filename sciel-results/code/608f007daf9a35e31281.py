"""
Optimized program for OneMax task.
Constructs a 100-bit binary string maximizing the number of ones.

Strategy:
The OneMax problem with N=100 has a known global optimum: a string of 100 '1's.
Instead of running an optimization algorithm, we precompute this optimal string
at module load time. The construct() function then returns this precomputed
constant in constant time O(1), ensuring minimal latency and guaranteed optimality.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN_SECONDS = 5.0

# Precompute the optimal solution once at module load time.
# This is the direct construction: all bits set to '1'.
# Using simple string multiplication is faster and lighter than NumPy for this specific case.
_OPTIMAL = "1" * N


def construct():
    """
    Return a binary string of length N maximizing the number of ones.

    For the OneMax problem with N=100, the global optimum is the string
    of 100 ones. This function returns that precomputed string.

    An explicit deadline guard is included per the task contract.
    """
    # Deadline guard: track elapsed time to ensure we return before the limit.
    # Although this function is O(1), we maintain the guard structure as directed.
    t0 = time.time()
    
    # The optimal solution is the precomputed all-ones string.
    result = _OPTIMAL
    
    # Safety check: if we are too close to the deadline, return immediately.
    elapsed = time.time() - t0
    if elapsed + SAFETY_MARGIN_SECONDS >= DEADLINE_SECONDS:
        return result
        
    return result
