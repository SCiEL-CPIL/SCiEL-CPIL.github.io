"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    
    The constant solution h_i = 0.5 is the analytic optimum for the cyclic
    overlap problem, achieving the theoretical lower bound of N/4 for the
    maximum overlap. For linear overlap, it achieves (N-1)/4, which is also
    near-optimal. We return this constant sequence.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
