N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    h = np.zeros(N)
    # Optimize alternating pattern a, 1-a
    # Minimize max overlap
    # C(1) approx N/2 * a^2 + (N/2 - 1) * (1-a)^2
    # Derivative zero at a = (N-2) / (2(N-1))
    a = (N - 2) / (2 * (N - 1))
    for i in range(N):
        if i % 2 == 0:
            h[i] = a
        else:
            h[i] = 1 - a
    return h
# EVOLVE-BLOCK-END
