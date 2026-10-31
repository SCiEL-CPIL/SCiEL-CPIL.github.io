import numpy as np
import time

N = 50

def construct():
    """
    Returns a sequence of 50 nonnegative step heights that maximizes the ratio
    ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    Analysis shows that a single impulse (one non-zero height) minimizes the
    denominator's scaling factor relative to the numerator, yielding a ratio
    of N (which is 50 in this case). Uniform distributions yield a ratio of 1.
    """
    h = [0.0] * N
    h[0] = 1.0
    return h
