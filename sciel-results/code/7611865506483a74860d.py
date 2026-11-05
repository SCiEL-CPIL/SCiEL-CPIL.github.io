"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The objective is OneMax: maximize the number of '1' bits in a binary string of length N.
    Since there are no constraints preventing bits from being '1', the optimal solution
    is a string consisting entirely of '1's.
    
    Returns:
        A string of length N containing only '1's.
    """
    return "1" * N
# EVOLVE-BLOCK-END
