"""
OneMax direct binary construction (100 bits).

Returns the global optimum for the OneMax problem: a 100-bit string
consisting entirely of '1's. This maximizes the objective function
(count of ones) to its theoretical maximum of 100.
"""

N = 100
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN_SECONDS = 5.0

# Precompute the optimal solution once at module load time.
# This is the direct construction: all bits set to '1'.
_OPTIMAL = "1" * N


def construct():
    """
    Return a binary string of length N maximizing the number of ones.

    For the OneMax problem with N=100, the global optimum is the string
    of 100 ones. This function returns that precomputed string.

    An explicit deadline guard is included per the task contract.
    """
    import time

    t0 = time.time()

    # The optimal solution for OneMax is the all-ones string.
    result = _OPTIMAL

    # Deadline guard: ensure we return well before the limit.
    # Since the operation is O(1), this check is purely for compliance.
    if time.time() - t0 + SAFETY_MARGIN_SECONDS >= DEADLINE_SECONDS:
        return result

    return result
