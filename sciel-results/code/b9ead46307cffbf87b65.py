"""
Solves the Erdos minimum overlap problem.
Objective: Minimize max_k sum_i h[i] * (1 - h[(i-k)%N])
Subject to: 0 <= h[i] <= 1, sum(h) = N/2.

Analytical Derivation:
Let A[k] be the circular autocorrelation of h: A[k] = sum_i h[i]h[(i-k)%N].
The overlap at shift k is O[k] = sum_i h[i] - A[k] = N/2 - A[k].
We want to minimize max_k O[k], which is equivalent to maximizing min_k A[k].
We know that sum_k A[k] = N * sum_i h[i]^2 / 2? No.
sum_k A[k] = sum_k sum_i h[i]h[i-k] = sum_i h[i] (sum_k h[i-k]) = sum_i h[i] (N/2) = N^2/4.
The average autocorrelation is (N^2/4) / N = N/4.
Since the average of A[k] is N/4, the minimum value of A[k] cannot exceed N/4.
Therefore, max_k O[k] = N/2 - min_k A[k] >= N/2 - N/4 = N/4.
The lower bound for the maximum overlap is N/4.
This bound is achieved if and only if A[k] is constant for all k, i.e., A[k] = N/4.
A[k] is constant iff the power spectral density |H[k]|^2 is a delta function at k=0.
This implies H[k] = 0 for all k != 0.
This implies h[i] is constant for all i.
Given sum(h) = N/2, the only constant solution is h[i] = 0.5.
Thus, h[i] = 0.5 is the unique global optimum.
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the optimal sequence of step heights.
    Returns a numpy array of length N with values 0.5.
    """
    # The unique global optimum is the uniform sequence
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
