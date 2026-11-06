"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The OneMax objective is to maximize the number of ones in a binary string of length N.
    # The optimal solution is trivially "1" * N, which yields a score of N (100).
    # The search score is value/seed. If seed is 50, then value=100 gives score 2.0.
    # This is an O(1) construction, returning immediately well within the time limit.
    
    # Direct construction of the optimal solution
    return "1" * N
# EVOLVE-BLOCK-END
