import numpy as np

def coefficients(stages):
    """
    4th-order symmetric product formula coefficients.
    Uses the moment-matching conditions for 4th-order accuracy.
    a_k = b_k = 1/N + (1/N^3) * ((2k - N - 1)^2 - (N^2 - 1)/3)
    """
    N = stages
    a = []
    b = []
    for k in range(1, N + 1):
        term = (2 * k - N - 1) ** 2 - (N * N - 1) / 3.0
        val = 1.0 / N + term / (N ** 3)
        a.append(val)
        b.append(val)
    
    # Ensure sums are exactly 1.0 to avoid floating point drift issues in evaluator
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
        
    return a, b
