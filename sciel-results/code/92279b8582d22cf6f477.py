# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import mobius


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    # Use the Möbius function mu(k) for k in [1, K].
    # The identity sum_{d|n} mu(d) = 1 if n=1 else 0 implies
    # sum_{k=1}^{floor(x)} mu(k) * floor(x/k) = 1 for all x >= 1.
    # This makes M(x) = 1 exactly, which is <= 1.0001, so feasible.
    # The objective a = -sum mu(k) log(k) / k converges to 1 as K -> infinity.
    
    K = 200000  # Maximum allowed key
    
    # Compute Möbius function up to K
    mu = mobius(np.arange(1, K + 1))
    
    # Filter out zeros to keep the domain small (mu(k) != 0 only for square-free k)
    nonzero_mask = mu != 0
    domain = np.arange(1, K + 1)[nonzero_mask].astype(np.int64)
    values = mu[nonzero_mask].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
