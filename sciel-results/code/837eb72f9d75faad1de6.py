"""Improved program: Delta-like distribution to maximize the ratio."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For h = [1, 0, ..., 0], R = 1.
    # For h = [1, ..., 1], R = 1/N = 0.02.
    # The seed score is 1.0, corresponding to R=0.02 (assuming normalization or scaling).
    # We aim for R=1.
    
    # Use a small epsilon to ensure strict positivity if required,
    # though "nonnegative" allows 0.
    eps = 1e-9
    h = [eps] * N
    h[0] = 1.0 - (N - 1) * eps
    
    return h
# EVOLVE-BLOCK-END
