"""
Spectral optimization for Erdos minimum overlap.
The problem is to minimize max_k sum_i h_i (1-h_{i+k}) subject to sum h_i = N/2, h_i in [0,1].
This is equivalent to maximizing min_k C_h(k), where C_h is the circular autocorrelation.
We know sum_k C_h(k) = (sum h_i)^2 = N^2/4.
Thus avg C_h(k) = N/4.
min C_h(k) <= N/4.
Therefore max_k O_k >= N/4.
The constant sequence h_i = 1/2 achieves C_h(k) = N/4 for all k, so max O_k = N/4.
Thus, the constant sequence is the global optimum.
"""

import numpy as np

N = 100

def construct():
    return np.full(N, 0.5)
