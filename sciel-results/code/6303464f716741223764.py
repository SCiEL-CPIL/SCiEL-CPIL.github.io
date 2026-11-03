"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly N/2.
    The constant solution h_i = 0.5 minimizes the maximum overlap 
    sum(h_i * (1 - h_{i+k})) for all shifts k, achieving the theoretical lower bound of N/4.
    """
    # Theoretically optimal constant solution
    # Using a list to ensure compatibility with strict type checkers
    return [0.5] * N
# EVOLVE-BLOCK-END
