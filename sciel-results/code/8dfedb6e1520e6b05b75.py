"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

N = 100
DEADLINE = 150  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction: OneMax on 100 bits.
    The optimal solution is a string of 100 '1's.
    Includes a deadline guard as per directive.
    """
    # Deadline guard: if we are too close to the wall-clock limit, return
    # the precomputed optimal string immediately.
    # This is extremely fast, but the guard ensures compliance with the
    # wall-clock constraint if this function were ever called in a loop
    # or context where time matters.
    elapsed = time.time() - START_TIME
    if elapsed > DEADLINE - 10:
        return "1" * N

    # Construct the optimal 100-bit string.
    # Direct string multiplication is O(N) and extremely fast in Python.
    return "1" * N
# EVOLVE-BLOCK-END
