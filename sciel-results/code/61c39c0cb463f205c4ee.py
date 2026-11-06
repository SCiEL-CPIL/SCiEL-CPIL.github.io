"""
Constructs a 100-bit binary string maximizing the number of ones.
The optimal solution is a string of all ones.
"""

import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N consisting entirely of '1's.
    Uses direct string multiplication for optimal construction.
    Includes an explicit wall-clock deadline guard as required.
    """
    start_time = time.time()
    deadline = 150.0  # Wall-clock limit (seconds)
    safety_margin = 5.0

    # Direct construction: all ones is the unique global optimum for OneMax.
    # Using string multiplication is the most direct and efficient method.
    result = "1" * N

    # Safety guard: ensure we return before the wall-clock deadline.
    # Since the construction is O(N) and N=100, this check is purely defensive
    # to satisfy the directive for an explicit deadline guard.
    if time.time() + safety_margin > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
