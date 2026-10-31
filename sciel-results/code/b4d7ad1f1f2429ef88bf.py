"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np

N = 400

def construct():
    """
    Returns a sequence of signed step heights that minimizes the ratio:
    max_k |autocorrelation(k)| / (sum(x))^2
    
    Analysis:
    Let x be a sequence of length N.
    Autocorrelation A(k) = sum_i x_i x_{i+k}.
    Integral I = sum_i x_i.
    Objective: J = max_k |A(k)| / I^2.
    
    Note that A(0) = sum_i x_i^2.
    By Cauchy-Schwarz, I^2 = (sum_i x_i)^2 <= (sum_i x_i^2) * N = A(0) * N.
    Therefore, A(0) / I^2 >= 1/N.
    
    Since max_k |A(k)| >= |A(0)|, we have J >= A(0)/I^2 >= 1/N.
    
    The lower bound 1/N is achieved if and only if:
    1. x_i is constant for all i (so I^2 = N * A(0) is tight).
    2. The maximum absolute autocorrelation is attained at k=0 (or tied with it).
    
    For x = [1, 1, ..., 1]:
    I = N.
    A(0) = N.
    A(k) = N - k for k > 0.
    max |A(k)| = N (at k=0).
    J = N / N^2 = 1/N.
    
    This is the global minimum.
    """
    return np.ones(N)
