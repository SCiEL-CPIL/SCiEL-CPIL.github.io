"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective with no constraints has a trivial optimum:
    all bits set to 1. This returns that string directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
