"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    The metric is max(|autoconv|) / (sum(signs))^2.
    For binary signs {-1, 1}, autoconv[0] = N and sum(signs) <= N.
    The metric is minimized when sum(signs) is maximized (all 1s or all -1s),
    giving a value of N / N^2 = 1/N.
    This is the global minimum.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
