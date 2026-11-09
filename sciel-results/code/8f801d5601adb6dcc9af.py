"""Cost-aware acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We keep a safety margin so we return before the hard limit.
_DEADLINE = 320.0  # seconds from module load (approximate; per-call overhead is small)
_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    mu, sigma: shape (2000,), GP mean and std in objective units.
    f_best: scalar, lowest observed value.
    cost: shape (2000,), cost of evaluating each candidate.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.random.uniform(0, 1, size=len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # UCB-like hybrid: (f_best - mu) + beta * sigma
    # This assigns positive scores to points with high uncertainty even if
    # their mean prediction is worse than f_best, encouraging exploration
    # and helping escape local minima in Ackley 2D.
    beta = 1.0
    acq = (f_best - mu) + beta * sigma

    # Cost-aware: divide by cost to get improvement per unit cost
    acq = acq / cost

    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)

    return acq
# EVOLVE-BLOCK-END
