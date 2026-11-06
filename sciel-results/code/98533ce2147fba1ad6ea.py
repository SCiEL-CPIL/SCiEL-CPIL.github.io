"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a set of 12 distinct integers in [0, 128] that minimizes the size
    of the largest Sidon subset.
    
    The arithmetic progression {0, 1, ..., 11} is the densest possible set of 12
    distinct integers. Dense sets have the most additive collisions, which
    minimizes the size of the largest Sidon subset. For n=12, the largest Sidon
    subset of an AP is known to be small (likely 4). Sparse sets allow for
    larger Sidon subsets.
    """
    # The densest set of 12 distinct integers in [0, 128] is the AP with step 1.
    return list(range(12))
# EVOLVE-BLOCK-END
