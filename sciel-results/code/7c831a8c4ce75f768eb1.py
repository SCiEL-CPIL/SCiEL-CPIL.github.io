"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(arr):
    """
    Compute the C3' metric: max|autoconv| / (sum|arr|)^2.
    Lower is better.
    """
    c = np.convolve(arr, arr, mode='full')
    max_abs = np.max(np.abs(c))
    integral = np.sum(np.abs(arr))
    if integral == 0:
        return np.inf
    return max_abs / (integral * integral)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin from 150s limit
    
    # Initial best: constant sequence
    # f = 1, max|c| = N, integral = N, score = N/N^2 = 1/N
    best_seq = np.ones(N)
    best_score = _score(best_seq)
    
    # Since 1/N is the theoretical minimum by Cauchy-Schwarz (c(0) = sum f^2 <= (sum|f|)^2, so c(0)/(sum|f|)^2 <= 1, wait...
    # Let's re-verify the bound.
    # Metric: M = max_k |sum_i f_i f_{i+k}| / (sum_i |f_i|)^2
    # For k=0, term is sum f_i^2.
    # By Cauchy-Schwarz: (sum |f_i|)^2 <= N * sum f_i^2. So sum f_i^2 >= (sum|f_i|)^2 / N.
    # Thus M >= ( (sum|f_i|)^2 / N ) / (sum|f_i|)^2 = 1/N.
    # Equality holds when |f_i| is constant.
    # If f_i = 1, sum f_i^2 = N, (sum|f_i|)^2 = N^2. M = N/N^2 = 1/N.
    # So 1/N is the global minimum.
    
    # Therefore, any non-constant sequence will have a score >= 1/N.
    # The constant sequence achieves the lower bound.
    # There is no point in searching.
    
    return best_seq

# EVOLVE-BLOCK-END
