"""Optimal program for the suite task; returns the theoretical minimum overlap solution."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of step heights h in [0, 1] such that sum(h) = N/2.
    The solution h_i = 0.5 for all i minimizes the maximum overlap of h with shifted 1-h.
    
    Proof:
    Let O_k be the overlap at shift k: O_k = sum_i h_i (1 - h_{i+k}).
    Sum_{k=0}^{N-1} O_k = Sum_{k=0}^{N-1} Sum_{i=0}^{N-1} (h_i - h_i h_{i+k})
    = N * Sum_i h_i - Sum_i h_i * Sum_k h_{i+k}
    Given Sum_i h_i = N/2, and Sum_k h_{i+k} = N/2 (cyclic shift sum is constant),
    Sum_{k} O_k = N(N/2) - Sum_i h_i (N/2) = N^2/2 - (N/2)(N/2) = N^2/2 - N^2/4 = N^2/4.
    The average overlap is (N^2/4) / N = N/4.
    Since Max(O_k) >= Average(O_k), the minimum possible maximum overlap is N/4.
    The constant sequence h_i = 0.5 yields O_k = Sum 0.5 * 0.5 = N/4 for all k.
    Thus, it achieves the global minimum.
    """
    h = np.full(N, 0.5, dtype=np.float64)
    return h
# EVOLVE-BLOCK-END
