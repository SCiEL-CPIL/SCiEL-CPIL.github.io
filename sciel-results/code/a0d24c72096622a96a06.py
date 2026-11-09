"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def priority(vector, n):
    # Calculate the sum of squared deviations from the mean (1).
    # This measures how "balanced" the vector is.
    # Lower deviation means more balanced (closer to all 1s or mix of 0,1,2).
    # We want to prioritize balanced vectors, so we return negative sum of squares.
    s = 0
    for v in vector:
        d = v - 1
        s += d * d
    return -s
# EVOLVE-BLOCK-END
