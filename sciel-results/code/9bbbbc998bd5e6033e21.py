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

    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB-like hybrid: (f_best - mu) + beta * sigma
    # This assigns positive scores to points with high uncertainty even if
    # their mean prediction is worse than f_best, encouraging exploration
    # and helping escape local minima in Ackley 2D.
    
    # Adaptive beta based on uncertainty relative to improvement scale
    sigma_med = np.median(sigma)
    improvement_scale = f_best - np.min(mu)
    if improvement_scale < 1e-9:
        improvement_scale = 1e-9
    
    # Larger beta when uncertainty is high relative to improvement
    beta = 1.5 * np.sqrt(np.log(1 + sigma_med / (improvement_scale + 1e-9)))
    beta = np.clip(beta, 0.8, 3.0)
    
    ucb_imp = np.maximum(f_best - mu + beta * sigma, 0)

    # Blend EI and UCB: trust UCB more when uncertainty is high
    w_ucb = sigma_med / (sigma_med + improvement_scale + 1e-9)
    w_ucb = np.clip(w_ucb, 0.3, 0.7)
    
    acq = (1 - w_ucb) * ei + w_ucb * ucb_imp

    # Cost-aware: divide by cost to get improvement per unit cost
    # Slightly amplify cost sensitivity when exploring more
    gamma = 1.0 + 0.2 * w_ucb
    acq = acq / (cost ** gamma)

    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)

    return acq
# EVOLVE-BLOCK-END
