# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement with an exploration bonus, both normalized by cost.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 250.0  # 300s limit, keep 50s margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Ensure no NaN or Inf inputs propagate
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e10, neginf=1.0)

    # Safety: ensure sigma is positive (tiny epsilon)
    sigma = np.maximum(sigma, 1e-9)

    # Safety: ensure cost is positive
    cost = np.maximum(cost, 1e-9)

    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure EI is non-negative (it should be, but numerical issues can occur)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: EI divided by cost
    cost_aware_ei = ei / cost

    # Exploration bonus: favor points with higher uncertainty (sigma)
    # This helps escape local optima in rastrigin
    # Scale the bonus to be comparable to the EI term
    # The typical EI value is on the order of sigma, so we scale by sigma
    # Use a small coefficient to balance exploration
    exploration_bonus = 0.05 * sigma / cost

    # Combine: cost-aware EI + exploration bonus
    scores = cost_aware_ei + exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
