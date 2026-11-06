"""OneMax direct binary construction (100 bits).

This module implements a deterministic construction of the optimal solution
for the OneMax problem, which requires maximizing the number of ones in a
binary string of length N. The optimal solution is a string of all ones.
"""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the optimal 100-bit binary string for the OneMax objective.
    
    The objective is to maximize the count of '1' bits. The theoretical maximum
    is N, achieved by a string consisting entirely of '1's. This function
    returns that string directly.
    """
    return "".join(['1'] * N)
# EVOLVE-BLOCK-END
