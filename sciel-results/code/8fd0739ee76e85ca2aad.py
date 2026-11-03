import numpy as np

N = 600

def construct():
    # The optimal function for minimizing the L-infinity norm of the autoconvolution
    # subject to a fixed L-1 norm and support is the uniform distribution.
    # We normalize the sum of heights to 1.
    return [1.0 / N] * N
