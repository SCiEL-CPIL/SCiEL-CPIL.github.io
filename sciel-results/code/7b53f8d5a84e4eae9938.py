import numpy as np

N = 600

def construct():
    # Create a zero array
    h = [0.0] * N
    # Set the center element to 1.0
    # This minimizes the autoconvolution peak (which is the sum of squares)
    # given the constraint that the maximum height is 1.
    # c_0 = sum(h_i^2) = 1.
    # For any other non-negative function with max height 1, c_0 >= 1.
    h[N // 2] = 1.0
    return h
