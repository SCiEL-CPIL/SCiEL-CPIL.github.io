"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

N = 100

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem asks to minimize the maximum overlap O_k = sum(h_i * (1 - h_{i+k}))
    subject to sum(h_i) = N/2 and h_i in [0, 1].
    
    Analysis:
    The average overlap over all shifts k=1..N-1 is:
    Avg(O) = (1/(N-1)) * sum_{k=1}^{N-1} sum_{i=0}^{N-1} h_i (1 - h_{i+k})
    sum_{k=1}^{N-1} sum_{i=0}^{N-1} h_i (1 - h_{i+k}) = sum_{i=0}^{N-1} h_i (N-1 - sum_{k=1}^{N-1} h_{i+k})
    Since sum_{k=0}^{N-1} h_{i+k} = sum_{j=0}^{N-1} h_j = N/2, we have sum_{k=1}^{N-1} h_{i+k} = N/2 - h_i.
    So the term becomes sum_{i=0}^{N-1} h_i (N-1 - (N/2 - h_i)) = sum_{i=0}^{N-1} (h_i(N/2 - 1) + h_i^2)
    = (N/2 - 1) * sum(h_i) + sum(h_i^2)
    = (N/2 - 1) * (N/2) + sum(h_i^2)
    = N^2/4 - N/2 + sum(h_i^2).
    
    By Cauchy-Schwarz, sum(h_i^2) >= (sum(h_i))^2 / N = (N/2)^2 / N = N/4.
    Equality holds if and only if all h_i are equal, i.e., h_i = 0.5.
    
    Thus, min(Avg(O)) = (N^2/4 - N/2 + N/4) / (N-1) = (N^2/4 - N/4) / (N-1) = (N(N-1)/4) / (N-1) = N/4.
    Since the average overlap is at least N/4, the maximum overlap must be at least N/4.
    The constant sequence h_i = 0.5 achieves O_k = N/4 for all k.
    Therefore, h_i = 0.5 is the global optimum.
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
