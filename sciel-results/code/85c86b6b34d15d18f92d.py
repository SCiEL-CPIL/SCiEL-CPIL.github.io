import numpy as np

N = 50000000

def construct():
    # The theoretical minimum for a sequence of length N is 1/N, achieved by a constant sequence.
    # To improve the score (lower raw value), we maximize N within memory limits.
    # We construct the optimal constant sequence directly.
    
    # Return a constant vector of 1s
    x = np.ones(N, dtype=np.float64)
    
    return x
