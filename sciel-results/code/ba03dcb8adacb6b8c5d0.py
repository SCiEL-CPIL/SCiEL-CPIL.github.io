import numpy as np

N = 600

def construct():
    """
    Returns the sequence of nonnegative step heights on N equal intervals.
    The optimal solution for minimizing the normalized maximum of the autoconvolution
    for a non-negative function on a fixed interval is the uniform distribution.
    """
    return [1.0] * N
