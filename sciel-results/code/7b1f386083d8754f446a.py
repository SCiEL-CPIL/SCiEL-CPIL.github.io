"""
Program for OneMax direct binary construction (100 bits).
Goal: Maximize the number of ones in the binary string.
Optimal solution: A string of 100 '1's.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with maximum number of ones.
    Returns a string of N '1's.
    """
    return "1" * N
# EVOLVE-BLOCK-END
