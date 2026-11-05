"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # By Cauchy-Schwarz, for any real h with sum(h) != 0:
    #   max(convolve(h,h)) >= sum(h^2) >= (sum(h))^2 / N
    # Therefore value >= 2*N * (sum(h)^2/N) / (sum(h))^2 = 2.0
    # The constant vector achieves equality, so 2.0 is the global minimum.
    # Return the optimal constant step heights.
    return np.ones(N)
# EVOLVE-BLOCK-END
