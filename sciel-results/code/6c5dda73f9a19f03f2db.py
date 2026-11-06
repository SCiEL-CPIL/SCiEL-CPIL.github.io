"""Direct OneMax construction: return the optimal 100-bit all-ones string."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with the maximum possible number of ones."""
    # The OneMax objective is maximized by setting every bit to 1.
    # Construct the optimal solution in O(n) time with no search required.
    return "1" * N
# EVOLVE-BLOCK-END
