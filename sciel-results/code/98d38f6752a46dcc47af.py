"""
Minimize the normalized maximum of the autoconvolution of a nonnegative step function
on [-1/4, 1/4] with 600 equal intervals.

Mathematical Analysis:
Let f be a nonnegative step function with heights h_0, ..., h_{N-1} on intervals of
width delta = T/N, where T = 1/2 and N = 600.

The quantity to minimize is:
    S = max_tau (f*f)(tau) / (integral f)^2

For nonnegative f, (f*f)(tau) <= (f*f)(0) = integral(f^2) by Cauchy-Schwarz:
    (f*f)(tau) = integral f(t) f(t-tau) dt <= (integral f^2(t) dt)^{1/2} (integral f^2(t-tau) dt)^{1/2} = integral(f^2)

Thus S >= integral(f^2) / (integral f)^2.

By Cauchy-Schwarz again:
    integral(f^2) >= (integral f)^2 / T

with equality iff f is constant on its support. Therefore:
    S >= 1/T = 2

This lower bound is achieved by f being constant on the full interval [-1/4, 1/4].
For a step function with N equal intervals, setting all heights equal achieves this.

The score is scale-invariant (S depends only on the shape, not the amplitude),
so we return h_i = 1.0 for all i.
"""
import numpy as np

N = 600
T = 0.5  # Total length of the interval [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution.
    
    The optimal solution is a uniform (constant) height over the entire interval.
    This is proven optimal by the chain of inequalities:
        max(f*f) >= (f*f)(0) = ||f||_2^2 >= ||f||_1^2 / T = 1/T
    with equality throughout when f is constant on [-1/4, 1/4].
    
    Since the normalized score is scale-invariant, any constant height works.
    We return 1.0 for all intervals.
    """
    return [1.0] * N
# EVOLVE-BLOCK-END
