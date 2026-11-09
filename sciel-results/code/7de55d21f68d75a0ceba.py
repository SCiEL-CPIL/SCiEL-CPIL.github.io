import time
import sys

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    The edges are ordered lexicographically: (0,1), (0,2), ..., (0,15), (1,2), ..., (14,15).
    Uses the coloring c(u, v) = min(u, v) for u < v.
    This is a valid Gallai coloring with N-1 = 15 distinct colors.
    """
    colors = []
    for i in range(N):
        for j in range(i + 1, N):
            # Color is min(i, j) which is i since i < j
            colors.append(i)
    return colors
# EVOLVE-BLOCK-END
