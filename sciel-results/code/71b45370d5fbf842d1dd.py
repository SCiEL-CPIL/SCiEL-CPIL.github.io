"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal height configuration.
    
    Analysis:
    The objective is to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Since ||h*h||_inf >= (h*h)[0] = ||h||_2^2, the ratio R is always <= 1.
    The maximum value of 1 is achieved when the peak of the autoconvolution
    occurs at the zero lag (or is equal to the zero lag value).
    
    The uniform distribution h_i = 1/N achieves this:
    ||h||_1 = 1
    ||h||_2^2 = 1/N
    (h*h)_k is a triangle wave peaking at N/N^2 = 1/N.
    Thus R = (1/N) / (1 * 1/N) = 1.
    
    Similarly, a delta function (one non-zero element) achieves R=1.
    
    We return the uniform distribution as it is robust and simple.
    """
    return np.ones(N) / N
# EVOLVE-BLOCK-END
