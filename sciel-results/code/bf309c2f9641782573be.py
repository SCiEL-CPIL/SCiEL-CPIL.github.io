"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # By Cauchy-Schwarz, sum(x_i^2) >= (sum x_i)^2 / N for any real sequence.
    # The autocorrelation at lag 0 is sum(x_i^2), so max|autocorr| >= (sum x_i)^2 / N.
    # Therefore, max|autocorr| / (sum x_i)^2 >= 1/N.
    # Equality holds iff all x_i are equal.
    # Thus, the constant sequence achieves the theoretical minimum of 1/N.
    # Return a constant sequence to achieve the optimal score.
    x = np.ones(N)
    return x
# EVOLVE-BLOCK-END
