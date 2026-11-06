"""
OneMax (100 bits) direct binary construction.
The objective is to maximize the count of '1' characters in a 100-bit string.
The global optimum is the all-ones string, which yields a score of 100.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    # OneMax is separable: f(x) = sum_i x_i, x_i in {0,1}.
    # The unique global maximizer sets every bit to 1.
    # We build the optimal string directly, with an explicit deadline guard.
    t0 = time.monotonic()
    # Construct the 100-bit all-ones string
    opt = "1" * N
    # Safety margin: return fallback if we somehow exceed the deadline
    if (time.monotonic() - t0) > (DEADLINE_SECONDS - 0.5):
        return "0" * N
    return opt
# EVOLVE-BLOCK-END
