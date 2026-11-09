"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for OneMax is all ones.
    """
    # The optimal solution for OneMax is a string of all '1's.
    # We return this directly to maximize the score.
    return "1" * N
# EVOLVE-BLOCK-END
