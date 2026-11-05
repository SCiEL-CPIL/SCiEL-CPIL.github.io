"""Optimized program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The objective function is bounded below by 2.0 for any real sequence h
    # where sum(h) != 0, due to the Cauchy-Schwarz inequality:
    # max(autoconv) >= autoconv[0] = sum(h^2) >= (sum(h))^2 / N
    # Thus value >= 2 * N * (sum(h)^2 / N) / (sum(h))^2 = 2.0
    # The constant vector achieves this bound exactly.
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
