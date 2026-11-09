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
    Uses a 128-bit integer constructed via left-shifts, then extracts
    the lower 100 bits using format(..., '0100b').
    """
    # Deadline guard: if we are too close to the wall-clock limit, return
    # the precomputed optimal string immediately.
    elapsed = time.time() - START_TIME
    if elapsed > DEADLINE - 10:
        return "1" * N

    # Construct a 128-bit integer with all bits set via repeated left-shifts.
    # Start with 1, left-shift 127 times to get 2^127, then add 1 to get 2^127 + 1.
    # Actually, simpler: build (1 << 128) - 1 by starting from 1 and shifting.
    val = 1
    for _ in range(127):
        val = (val << 1) | 1
    # Now val = (1 << 128) - 1, a 128-bit all-ones integer.
    # Extract the lower 100 bits as a binary string.
    return format(val & ((1 << N) - 1), '0{}b'.format(N))

# EVOLVE-BLOCK-END
