"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective is to maximize the number of 1s in a binary string.
    The global optimum is a string of all 1s, which yields a score of N.
    This is achieved in constant time with no constraints other than length.
    """
    # The optimal solution for OneMax is all ones
    return "1" * N
# EVOLVE-BLOCK-END
