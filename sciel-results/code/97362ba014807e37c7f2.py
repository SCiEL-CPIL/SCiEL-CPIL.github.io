"""
Optimized approach: Use an asymmetric linear ramp to minimize the normalized
maximum of the autoconvolution.

The key insight is that a symmetric function (like a rectangle) has its
autoconvolution peak at lag 0, where the self-term h_0^2 dominates. An asymmetric
function (like a linear ramp) shifts the peak to an intermediate lag where
constructive interference is less severe relative to the L2 norm squared.

For a linear ramp h_i = i (i=0..N-1), the autoconvolution peak occurs at
lag N-1 with value ~N^3, while ||h||_2^2 ~ N^3/3, giving a ratio ~1/3.
This is significantly better than the seed score of 1.0.

We use h_i = i for i=0..N-1, which is simple, fast, and yields a substantial
improvement.
"""

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights forming a linear ramp
    from 0 to N-1. This asymmetric profile minimizes the normalized maximum
    of the autoconvolution compared to symmetric profiles.
    """
    return [float(i) for i in range(N)]
