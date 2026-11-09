"""Optimized OneMax constructor: direct binary construction maximizing ones."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits.
    The global optimum is a string of length N consisting entirely of '1's.
    This function directly constructs this optimal string.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    # The optimal solution for OneMax is all ones.
    # This is instantaneous and requires no time management.
    return "1" * N
# EVOLVE-BLOCK-END
