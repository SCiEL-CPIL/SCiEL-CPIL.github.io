"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_div(a, b, eps=1e-12):
    return np.divide(a, b, out=np.zeros_like(a), where=b > eps)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition that balances expected improvement against cost,
    with a dynamic exploration threshold and safety margin for the deadline.
    """
    t0 = time.time()
    deadline = 330.0 - 30.0  # safety margin

    n = len(mu)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Dynamic xi: lower early to explore more, higher later to exploit
    # cost starts at ~5 (5 initial points) and goes up to ~30
    # Map cost to xi in [0.01, 0.08]
    cost_min = 5.0
    cost_max = 30.0
    cost_frac = np.clip((cost - cost_min) / (cost_max - cost_min), 0.0, 1.0)
    # Use the median cost as a proxy for "how far along" we are
    median_cost_frac = np.median(cost_frac)
    xi = 0.01 + 0.07 * median_cost_frac  # 0.01 to 0.08

    # Expected Improvement with xi
    improvement = f_best - mu - xi
    z = _safe_div(improvement, sigma)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost normalization: soft penalty instead of hard division
    # cost is in [1, 2], so 1 + cost_factor where cost_factor is small
    cost_factor = 0.5  # scale factor
    cost_penalty = 1.0 + cost_factor * (cost - 1.0)  # ranges from 1.0 to 1.5
    score = ei / cost_penalty

    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local minima in the 8D landscape
    uncertainty_bonus = 0.1 * sigma / (sigma.max() + 1e-9)
    score = score + uncertainty_bonus

    # Check deadline
    if time.time() > deadline:
        # Return the current best result
        return score

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
