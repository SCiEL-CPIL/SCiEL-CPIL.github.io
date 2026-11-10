# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
    The objective is a = -sum f(k) log(k)/k.
    The constraint is M(x) = sum f(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
    Let f(k) = -g(k) for k > 1 with g(k) >= 0.
    Then M(x) = sum_{k>1} g(k) {x/k}.
    To maximize a = sum g(k) log(k)/k subject to max_x sum g(k) {x/k} <= 1.
    Analysis shows that k=2 provides the highest ratio of objective coefficient to constraint usage.
    For k=2, max {x/2} = 0.5. Coeff log(2)/2.
    Setting g(2) = 2 gives M(x) <= 2 * 0.5 = 1.
    f(2) = -2.
    f(1) is adjusted to 1 to satisfy sum f(k)/k = 0.
    This yields a = log(2) ~ 0.693.
    """
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([1.0, -2.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
