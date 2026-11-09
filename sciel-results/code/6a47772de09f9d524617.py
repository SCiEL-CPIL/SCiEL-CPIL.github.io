"""Construct a 100-bit binary string maximizing ones (OneMax)."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Return the globally optimal 100-bit string for OneMax.
    The objective is to maximize the count of '1' bits, so the unique
    optimum is a string of N ones. We build it via list join for
    clarity and deterministic output.
    """
    return ''.join(['1'] * N)
# EVOLVE-BLOCK-END
