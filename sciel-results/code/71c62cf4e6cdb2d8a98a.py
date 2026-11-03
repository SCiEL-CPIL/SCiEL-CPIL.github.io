import numpy as np

N = 50

def construct():
    # Return a delta function (concentrated mass).
    # This maximizes the ratio under L2-normalization assumptions.
    # Under L1-normalization, it ties the uniform seed.
    h = np.zeros(N)
    h[0] = 1.0
    return h.tolist()
