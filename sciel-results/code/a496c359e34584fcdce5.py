import numpy as np

N = 600

def construct():
    # Create a step function with alternating 1s and 0s
    # This maximizes the number of disjoint blocks (K=300)
    # assuming the score is max(g) / (||f||_1 * ||f||_inf)
    h = np.zeros(N)
    h[0::2] = 1.0
    return list(h)
