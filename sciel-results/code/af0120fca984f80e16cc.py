"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Direct construction of the global optimum: all 100 bits set to 1.
    # The OneMax objective is to maximize the count of 1s in a binary string of length N.
    # The maximum possible count is N, achieved uniquely by the string of N ones.
    # This is a deterministic, O(N) construction with no search required.
    # Include a deadline guard for compliance with the wall-clock constraint,
    # though execution is instantaneous for N=100.
    deadline = time.time() + 10.0
    optimal = "1" * N
    # Verify length and content as a sanity check (optional, low cost)
    assert len(optimal) == N
    assert all(c == '1' for c in optimal)
    return optimal
# EVOLVE-BLOCK-END
