"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal solution for the Erdos minimum overlap problem.
    
    Analysis:
    The objective is to minimize the maximum overlap O(t) = (1/N) * sum_i h[i] * (1 - h[i+t]).
    O(t) = (1/N) * (sum_i h[i] - sum_i h[i]h[i+t]) = 0.5 - C(t)/N, where C(t) is the circular autocorrelation.
    Minimizing max_t O(t) is equivalent to maximizing min_t C(t).
    
    The sum of autocorrelations over all shifts is:
    sum_t C(t) = sum_t sum_i h[i]h[i+t] = sum_i h[i] (sum_t h[i+t]) = sum_i h[i] * (N/2) = (N/2) * (N/2) = N^2/4.
    The average value of C(t) is (N^2/4) / N = N/4.
    Therefore, min_t C(t) <= N/4.
    Consequently, max_t O(t) >= 0.5 - (N/4)/N = 0.5 - 0.25 = 0.25.
    
    This lower bound of 0.25 is achieved if and only if C(t) is constant for all t, which requires h[i] to be constant.
    Given the constraint sum(h) = N/2 and h[i] in [0,1], the constant solution is h[i] = 0.5 for all i.
    
    Thus, the uniform array of 0.5 is the unique global optimum.
    """
    N = 100
    h = np.full(N, 0.5)
    return h
# EVOLVE-BLOCK-END
