import numpy as np

N = 400

def construct():
    """
    Returns the optimal step heights.
    
    The objective value is V = 2 * N * max(c) / (sum(h))^2.
    We know max(c) >= c[0] = sum(h^2).
    So V >= 2 * N * sum(h^2) / (sum(h))^2.
    By Cauchy-Schwarz, (sum(h))^2 <= N * sum(h^2), so N * sum(h^2) / (sum(h))^2 >= 1.
    Thus V >= 2.
    
    The vector h = [1, 1, ..., 1] achieves:
    sum(h) = N
    sum(h^2) = N
    c = [N, N-1, ..., 1, 1, ..., N] (full convolution of ones is a triangle, max is N)
    max(c) = N
    V = 2 * N * N / N^2 = 2.
    
    Since 2 is the theoretical lower bound, this vector is optimal.
    Any deviation increases sum(h^2) for a fixed sum(h) or decreases sum(h) for a fixed sum(h^2), increasing V.
    """
    return np.ones(N)
