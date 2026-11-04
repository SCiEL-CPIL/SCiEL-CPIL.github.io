"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Create a signed step function with zero mean to minimize the L1 norm of the autocorrelation
    # The L1 norm of the autocorrelation is (integral of f)^2.
    # By making the sum of heights zero, we minimize this quantity.
    # An alternating sequence of 1s and -1s achieves this for even N.
    heights = np.ones(N)
    heights[1::2] = -1.0
    return heights
# EVOLVE-BLOCK-END
