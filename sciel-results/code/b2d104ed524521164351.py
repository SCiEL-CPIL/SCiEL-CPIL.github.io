"""
Optimized program: Returns the theoretically optimal signed step sequence for the given metric.
For a sequence of +/- 1, the objective is max|autoconv| / (sum h)^2.
The autoconvolution at lag 0 is sum(h^2) = N.
The sum is S.
Objective >= N / S^2.
Max S^2 is N^2 (when all h are same sign).
Min Objective is 1/N.
This is achieved by all 1s or all -1s.
"""
import numpy as np

N = 400

def construct():
    """
    Returns the optimal sequence of signed step heights.
    """
    # The all-ones sequence minimizes the ratio max|autoconv| / integral^2
    # for +/- 1 sequences.
    return np.ones(N)
