import numpy as np
import time

N = 100

def construct():
    """
    The theoretical lower bound for the maximum overlap C_k is 0.5.
    This is achieved when the autocorrelation of the deviation from 0.5 is zero for all lags k != 0,
    or simply when the sequence is constant at 0.5.
    For h_i = 0.5, C_k = 0.5 for all k.
    Any deviation from 0.5 will create negative autocorrelation peaks, increasing the maximum overlap.
    Therefore, the optimal solution is the constant sequence 0.5.
    """
    return np.full(N, 0.5)
