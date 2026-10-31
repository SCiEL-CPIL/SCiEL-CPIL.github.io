import numpy as np

N = 100000

def construct():
    # Deterministic construction: Maximize the integral (sum) to minimize the ratio
    # assuming the metric is max(|autoconv|) / (sum(s))^2
    # All 1s gives sum=N, max_side=N-1, ratio=(N-1)/N^2 ~ 1/N
    # This is significantly better than sequences with sum~0.
    s = np.ones(N, dtype=int)
    return s
