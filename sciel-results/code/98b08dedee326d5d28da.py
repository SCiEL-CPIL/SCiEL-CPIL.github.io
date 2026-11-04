import numpy as np

N = 600

def construct():
    """
    Returns a list of 600 nonnegative step heights on equal intervals of [-1/4, 1/4].
    
    The task is to minimize the normalized maximum of the autoconvolution.
    For a non-negative function f, the autoconvolution g(t) = (f*f)(t) has its maximum at t=0.
    g(0) = ∫ f(τ)^2 dτ = ||f||_2^2.
    
    The "normalized maximum" is likely max(g) / ||f||_2^2 or max(g) / (∫f)^2 or similar.
    If the normalization is by ||f||_2^2, the ratio is always 1.0 for any non-negative f.
    If the normalization is by (∫f)^2, the ratio is ||f||_2^2 / (∫f)^2, which is minimized by a constant function (ratio = 1.0).
    If the normalization is by ||f||_∞ * ||f||_1, or other combinations, the constant function is often optimal or near-optimal.
    
    Given the constraint that the function must be non-negative, the constant function (all heights equal) is a strong candidate for optimality,
    as it minimizes the concentration of energy, leading to the smallest possible peak in the autoconvolution relative to standard norms.
    
    We return a constant function with heights chosen such that the function is non-negative.
    The absolute scale doesn't matter for ratios, but we choose heights = 1.0 for simplicity.
    """
    # Return a list of 600 ones
    return [1.0] * N
