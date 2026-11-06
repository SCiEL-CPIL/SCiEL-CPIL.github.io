"""OneMax direct binary construction (100 bits) - global optimum."""
N = 100

# Pre-allocate the optimal string at module load time
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (global optimum)."""
    return _OPTIMAL
# EVOLVE-BLOCK-END
