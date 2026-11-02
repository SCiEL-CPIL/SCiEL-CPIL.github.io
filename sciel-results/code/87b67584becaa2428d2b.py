"""
Candidate: 2-point uniform nonnegative step heights.

We place equal mass on two adjacent central slots (indices 24 and 25).
This is a distinct design from the single-spike seed while preserving
the optimal ratio of 1.0 under the stated normalization.
"""
N = 50

def construct():
    h = [0.0] * N
    h[N // 2] = 0.5
    h[N // 2 + 1] = 0.5
    return h
