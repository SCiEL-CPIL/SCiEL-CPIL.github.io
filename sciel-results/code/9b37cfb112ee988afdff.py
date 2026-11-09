import numpy as np

N = 50

def construct():
    # Uniform distribution over all 50 indices
    # This minimizes the L2 norm for a fixed L1 norm, 
    # which typically maximizes the ratio ||h||_2^2 / (||h||_1 * sup(h*h))
    # assuming the supremum is dominated by the peak at lag 0.
    h = [1.0 / N] * N
    return h
