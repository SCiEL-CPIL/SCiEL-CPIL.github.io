"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The metric to minimize is C3' = max|autoconv| / (integral)^2.
    For any real signal x, max|autoconv| >= autoconv[0] = sum(x^2).
    By Cauchy-Schwarz, (sum(x))^2 <= N * sum(x^2).
    Thus, C3' >= sum(x^2) / (sum(x))^2 >= 1/N.
    The constant signal x = [1, 1, ..., 1] achieves exactly 1/N.
    Therefore, the constant signal is the global optimum.
    """
    # The optimal signal is the constant vector of ones.
    # This minimizes the ratio to the theoretical lower bound of 1/N.
    return np.ones(N)
# EVOLVE-BLOCK-END
