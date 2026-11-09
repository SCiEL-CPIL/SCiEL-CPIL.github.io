import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    
    For the OneMax problem with a linear objective function where each '1' contributes
    positively to the score, the global optimum is a string consisting entirely of '1's.
    Therefore, we directly construct this string.
    """
    # The optimal solution for OneMax is a string of all 1s.
    # This is O(N) to construct and returns the maximum possible score.
    return '1' * N
# EVOLVE-BLOCK-END
