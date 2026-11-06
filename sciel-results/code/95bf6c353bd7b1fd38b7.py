"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for the OneMax problem.
    The optimal solution is a string of all '1's, yielding a raw score of 100.
    """
    # Directly construct the optimal solution: all ones
    return "1" * N
# EVOLVE-BLOCK-END
