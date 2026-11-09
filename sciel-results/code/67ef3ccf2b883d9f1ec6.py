"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax 100-bit construction.
    The global optimum is a string of 100 '1's.
    We return this directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
