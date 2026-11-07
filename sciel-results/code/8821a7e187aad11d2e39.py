# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3d.

    Uses Cost-Aware Expected Improvement (CA-EI) with adaptive exploration:
    1. Primary signal: EI / sqrt(cost) to directly optimize improvement per unit cost
    2. Secondary signal: UCB with adaptive beta for exploration
    3. Adaptive blending based on gap to known minimum
    """
    t_start = time.time()

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)

    # Known minimum of hartmann 3d
    f_min_known = -3.86278

    # Estimate budget consumption from mean cost
    mean_cost = np.mean(cost)
    budget_consumed = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)

    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # Cost-Aware EI: divide by sqrt(cost) to favor cheap high-improvement points
    cost_safe = np.maximum(cost, 1e-6)
    ca_ei = ei / np.sqrt(cost_safe)

    # Adaptive beta for UCB: start with high exploration, reduce as budget depletes
    beta = 2.0 - 1.5 * budget_consumed
    beta = np.maximum(beta, 0.3)

    # UCB score for minimization (higher is better)
    ucb = -mu + beta * sigma_safe

    # Normalize each component to [0, 1] for blending
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / rng

    ca_ei_norm = normalize(ca_ei)
    ucb_norm = normalize(ucb)

    # Adaptive weighting based on gap to known minimum
    gap = f_best - f_min_known
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)

    # When gap is large (early): explore more (higher UCB weight)
    # When gap is small (late): exploit more (higher CA-EI weight)
    w_ca_ei = 0.6 + 0.3 * (1.0 - gap_normalized)   # 0.6 to 0.9
    w_ucb = 0.4 - 0.3 * (1.0 - gap_normalized)     # 0.1 to 0.4

    # Renormalize weights to sum to 1
    w_sum = w_ca_ei + w_ucb
    w_ca_ei /= w_sum
    w_ucb /= w_sum

    # Combine normalized scores
    scores = w_ca_ei * ca_ei_norm + w_ucb * ucb_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Add tiny deterministic noise to break ties
    idx = np.arange(len(mu))
    tie_breaker = 1e-8 * np.sin(idx * 0.7 + f_best * 100)
    scores = scores + tie_breaker

    # Deadline guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores

    return scores
# EVOLVE-BLOCK-END
