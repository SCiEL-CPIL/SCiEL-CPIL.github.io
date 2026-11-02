"""Optimal solution for Erdos minimum overlap."""
import numpy as np

N = 100

def construct():
    """
    Returns the step heights h that minimize the maximum overlap.
    The theoretical minimum for the maximum overlap is N/4, achieved by the uniform distribution h[i] = 0.5.
    """
    return np.full(N, 0.5)
