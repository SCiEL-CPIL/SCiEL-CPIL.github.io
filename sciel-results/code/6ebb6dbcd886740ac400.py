"""Rewritten program: Thompson-sampling acquisition with cost-awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.

    Samples from the posterior (mu + sigma * z) and scores candidates by
    the negative sampled objective divided by evaluation cost. This naturally
    balances exploration and exploitation without explicit hyperparameters.
    """
    t0 = time.time()

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)

    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # Thompson sampling: draw one posterior sample
    # Use a single standard normal draw vectorized over 2000 candidates
    z = np.random.standard_normal(mu.shape)
    sample = mu_safe + sigma_safe * z

    # We want to minimize, so lower sample is better.
    # Score = -sample / cost (higher score = more desirable to evaluate)
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    score = -sample / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        # Fallback: simple EI-like score divided by cost
        z_ei = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * 0.5 * (1 + np.tanh(z_ei / 2.0)) + sigma_safe * np.exp(-0.5 * z_ei * z_ei)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
