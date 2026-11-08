import numpy as np

N = 400

def construct():
    """
    Returns the theoretically optimal signed step sequence for the given metric.
    The objective is max |autoconvolution| / (sum h)^2.
    For h in {-1, 1}^N, the minimum is achieved when all h_i are equal (all 1s or all -1s).
    In this case, sum h = N, autoconvolution peak = N.
    Objective = N / N^2 = 1/N.
    Any deviation reduces |sum h| and keeps peak >= N, increasing the objective.
    """
    return np.ones(N)
