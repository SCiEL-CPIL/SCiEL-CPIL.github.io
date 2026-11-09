"""Optimized OneMax constructor: direct binary construction maximizing ones."""

def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits.
    The global optimum is a string of length 100 consisting entirely of '1's.
    This function directly constructs this optimal string.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    return "1" * 100
