"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a sequence of step heights in [0, 1] whose sum is exactly N/2,
    minimizing the maximum overlap with shifted (1-h).
    
    Strategy:
    1. Analytical derivation shows that for circular shifts, the minimum possible
       maximum overlap is N/4, achieved by the constant solution h[i] = 0.5.
    2. Proof:
       Let O(s) = sum_i h[i]*(1 - h[(i+s)%N]) = N/2 - C(s), where C(s) is the
       circular autocorrelation.
       The average of C(s) over all s is (sum h[i])^2 / N = (N/2)^2 / N = N/4.
       Since min_s C(s) <= average(C(s)) = N/4, we have max_s O(s) >= N/2 - N/4 = N/4.
       The constant solution h[i] = 0.5 gives C(s) = N/4 for all s, so
       max_s O(s) = N/4, which is optimal.
    3. Therefore, return the constant solution directly.
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
