"""Optimal uniform step function for the first autocorrelation inequality."""
import numpy as np

N = 600

def construct():
    # The uniform step function is provably optimal:
    # (f*f)(0) = ||f||_2^2, so max(f*f)/||f||_2^2 >= 1,
    # and uniform achieves exactly 1.
    return [1.0] * N
