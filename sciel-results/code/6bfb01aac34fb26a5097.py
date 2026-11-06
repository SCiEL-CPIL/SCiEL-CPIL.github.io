"""OneMax direct binary construction (100 bits) - global optimum."""

N = 100
# Pre-compute the optimal solution: a string of 100 '1's
_OPTIMAL_STR = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (global optimum)."""
    return _OPTIMAL_STR
# EVOLVE-BLOCK-END
