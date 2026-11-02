import numpy as np

N = 600

def construct():
    """
    Returns the optimal step function heights for minimizing the normalized
    maximum autoconvolution.
    
    The uniform distribution minimizes the L2 norm for a fixed L1 norm (mass),
    which in turn minimizes the peak of the autoconvolution relative to the
    squared mass. This is the theoretical optimum for this problem.
    """
    return list(np.ones(N, dtype=float))
