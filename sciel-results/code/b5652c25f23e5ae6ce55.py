"""Rewritten program: minimize peak autoconvolution of a nonnegative 600-interval step function on [-1/4,1/4]."""
import time
import numpy as np

N = 600
T = 0.5  # interval length: 1/4 - (-1/4)

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution.
    
    Key insight: For a nonnegative function f, the autoconvolution (f*f)(x) has its
    maximum at x=0 when f is symmetric, and (f*f)(0) = ||f||_2^2 (the L2 norm squared).
    By Cauchy-Schwarz, (f*f)(x) <= ||f||_2^2 for all x, so max(f*f) >= ||f||_2^2,
    with equality iff f is a delta-like peak or constant. For fixed L1 norm, the
    uniform distribution minimizes ||f||_2^2, and for uniform, max(f*f) = ||f||_2^2.
    
    However, let's verify: if f is constant on its support, say f = c on an interval
    of length L, then L1 = c*L = 1 (normalized), so c = 1/L. Then (f*f)(0) = c^2 * L = 1/L.
    And (f*f)(x) for |x| <= L is c^2*(L-|x|), so max is c^2*L = 1/L.
    To minimize 1/L, maximize L. The support is [-1/4, 1/4], so L = 1/2, giving 1/L = 2.
    
    But wait, the "normalized maximum" likely means max(f*f) / (L1)^2 = max(f*f) since L1=1.
    
    So the uniform distribution over the full interval [-1/4, 1/4] with height 2 gives
    max autoconvolution = 2. This should be optimal.
    
    Let me double-check: if we use a narrower support, say [-1/8, 1/8], L=1/4, c=4,
    max autoconvolution = c^2 * L = 16 * 1/4 = 4 > 2. So wider is better.
    
    The full interval is the best. Uniform height = 2 (since L1 = 2 * 0.5 = 1).
    """
    # Uniform distribution over [-1/4, 1/4]
    # L1 norm = sum(heights) * (T/N) = 1
    # height * N * (T/N) = height * T = 1
    # height = 1/T = 2
    h = 1.0 / T  # = 2.0
    return [h] * N
# EVOLVE-BLOCK-END
