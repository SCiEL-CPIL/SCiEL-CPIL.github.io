N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    """
    # Direct construction: all ones is the unique global optimum for OneMax.
    # No time tracking needed as execution is instantaneous.
    return "1" * N
# EVOLVE-BLOCK-END
