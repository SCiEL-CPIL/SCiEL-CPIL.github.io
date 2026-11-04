"""Rewritten program using an explicit construction of the optimal constant vector."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal sequence of signed step heights.
    
    The objective function to minimize is:
    S = max_k |c_k| / (sum x_i)^2
    
    Let c_k be the autoconvolution of x. c_0 = sum x_i^2.
    Since c_0 is one of the terms in the max, max_k |c_k| >= c_0.
    By Cauchy-Schwarz, (sum x_i)^2 <= N * sum x_i^2 = N * c_0.
    Therefore, S >= c_0 / (N * c_0) = 1/N.
    
    The constant vector x_i = 1 achieves:
    sum x_i = N
    (sum x_i)^2 = N^2
    c_0 = N, c_k = N-k for k>0.
    max |c_k| = N.
    S = N / N^2 = 1/N.
    
    Since the lower bound is achieved, the constant vector is the unique global minimizer
    (up to scaling and sign).
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
