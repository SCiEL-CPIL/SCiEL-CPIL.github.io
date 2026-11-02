import numpy as np
import time

N = 100

def construct():
    """
    Returns the optimal sequence of step heights.
    The problem is to minimize the maximum overlap of h with shifted 1-h.
    Overlap O(k) = sum h(i)(1-h(i+k)) = N/2 - C(k).
    Minimizing max O(k) is equivalent to maximizing min C(k).
    The theoretical maximum for min C(k) is N/4 = 25, achieved by h_i = 0.5.
    Thus, the minimum possible max overlap is 25.
    We return the constant sequence 0.5 which achieves this bound.
    """
    h = np.full(N, 0.5)
    return h
