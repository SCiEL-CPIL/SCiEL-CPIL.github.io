import numpy as np

N = 100

def construct():
    """
    Construct the step heights h.
    The optimal solution is the uniform distribution h_i = 0.5, which minimizes the maximum overlap.
    """
    return np.full(N, 0.5)
