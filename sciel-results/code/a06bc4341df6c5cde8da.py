"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Theoretical analysis: for any step function h with sum S = sum(h_i),
    # autoconvolution C(k) = sum_{i} h_i * h_{i+k}
    # Ratio = max_k |C(k)| / S^2
    # C(0) = sum(h_i^2) >= S^2 / N by Cauchy-Schwarz
    # So ratio >= 1/N, achieved by constant h_i = c
    # 
    # The constant function is optimal. Return it.
    # Use 1.0 for simplicity (scale-invariant).
    return np.ones(N)
# EVOLVE-BLOCK-END
