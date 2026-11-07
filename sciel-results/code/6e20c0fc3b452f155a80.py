import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Leave 50s for cleanup/other overhead
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    Uses a structural change: Thompson Sampling with cost-aware scaling.
    Instead of deterministic EI/UCB, we sample from the posterior and select
    points that balance predicted performance with cost efficiency.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: exploit current best
        return -mu.copy()

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Thompson Sampling: draw a single sample from the posterior for each candidate
    # This is equivalent to maximizing a random objective: -mu + sigma * z
    # where z ~ N(0,1). We use a fixed seed based on call count for reproducibility
    # within the call, but different across calls.
    rng = np.random.RandomState(_CALL_COUNT * 12345 + 67890)
    z = rng.standard_normal(len(mu))
    
    # Sampled objective: lower is better
    sampled_obj = mu - sigma * z  # equivalent to -(-mu + sigma*z)

    # Cost-aware adjustment: prefer cheaper points
    # Normalize cost to [0, 1] range for the current batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: exponential decay based on normalized cost
    # Higher cost -> lower score
    cost_penalty = np.exp(-2.0 * cost_norm)
    
    # Score: negative sampled objective (so higher is better) times cost penalty
    score = -sampled_obj * cost_penalty

    # Add a small exploration bonus based on uncertainty
    # This helps avoid getting stuck in local optima
    uncertainty_bonus = 0.1 * sigma
    
    # Final score
    final_score = score + uncertainty_bonus

    # Ensure all scores are finite
    score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
