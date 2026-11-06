"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The optimal solution for OneMax is the string of all ones.
    # Direct construction yields the maximum possible score immediately.
    return "1" * N
# EVOLVE-BLOCK-END
