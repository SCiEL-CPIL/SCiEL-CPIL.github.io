"""
Optimized program for OneMax task.
Directly constructs the global optimum: a string of 100 ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the global optimum for the OneMax problem.
    """
    # Direct construction of the optimal solution
    return "1" * N
# EVOLVE-BLOCK-END
