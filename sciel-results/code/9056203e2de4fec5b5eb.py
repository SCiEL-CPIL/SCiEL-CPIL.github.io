"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Since the objective is OneMax (maximize number of 1s) and there are no
    constraints mentioned other than the length being N=100, the optimal solution
    is simply a string of all 1s.
    """
    return "1" * N
# EVOLVE-BLOCK-END
