"""Optimized OneMax constructor: direct binary construction maximizing ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits in a binary string
    of length N. The global optimum is trivially a string of all '1's, which
    achieves a score of N (100 ones).
    
    This function returns the known global optimum directly. No iterative
    optimization is needed since the solution is known a priori.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    # The optimal solution for OneMax is all ones.
    # This takes O(N) time and O(N) space, which is negligible.
    return "1" * N
# EVOLVE-BLOCK-END
