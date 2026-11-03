import numpy as np
import time

N = 400

def construct():
    # The objective function is max|autoconv| / (sum h)^2.
    # Let S = sum(h).
    # The autoconvolution at lag 0 is C_0 = sum(h_i^2).
    # Since max|C_k| >= C_0, the ratio is >= C_0 / S^2.
    # By Cauchy-Schwarz inequality, (sum h_i)^2 <= N * sum(h_i^2).
    # Therefore, C_0 / S^2 >= 1/N.
    # The minimum possible value for the objective is 1/N.
    # This minimum is achieved if and only if all h_i are equal (h_i = c for all i).
    # Due to scale invariance, we can choose c = 1.
    # Thus, returning a vector of ones provides the optimal solution.
    
    return np.ones(N)
