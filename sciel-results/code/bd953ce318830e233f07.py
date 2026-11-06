"""OneMax direct binary construction (100 bits) - ANALYTICAL CONSTRUCTION approach."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy: Analytical Constructive Optimization.
    The global optimum for OneMax is the string of all '1's.
    This function returns that optimal string directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
