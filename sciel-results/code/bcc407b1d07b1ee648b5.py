"""Optimal program for the suite task."""
import numpy as np
import time

N = 400

def construct():
    # The constant sequence achieves the theoretical lower bound of 1/N.
    # Any deviation increases the ratio.
    return np.ones(N)
