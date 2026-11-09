"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the optimal 100-bit binary string.
    
    The objective is to maximize the number of '1's in a 100-bit string.
    The global optimum is a string of 100 '1's, which yields a score of 100.
    This function returns that string directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
