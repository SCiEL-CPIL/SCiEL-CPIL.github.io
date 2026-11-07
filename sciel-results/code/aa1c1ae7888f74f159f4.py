"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    of different cardinalities are distinct.
    Uses an arithmetic progression with difference 1 and starting value A
    chosen to satisfy the condition m_{k+1} > M_k for all k.
    """
    # Start with the known valid size 31 solution
    # S = {226, ..., 256}
    # Size 31
    start_val = 226
    size = 31
    return [start_val + i for i in range(size)]
# EVOLVE-BLOCK-END
