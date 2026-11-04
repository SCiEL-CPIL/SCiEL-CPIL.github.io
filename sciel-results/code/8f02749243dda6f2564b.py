import numpy as np

def construct():
    N = 600
    # The metric is min max_conv / integral_sq.
    # This is minimized by the uniform distribution.
    # f = [1, 1, ..., 1]
    return [1.0] * N
