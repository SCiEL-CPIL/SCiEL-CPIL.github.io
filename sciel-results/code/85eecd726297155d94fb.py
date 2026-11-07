"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson sampling acquisition for Bayesian optimization of cosine8 8d.
    
    Structural change from EI/UCB blend to posterior sampling:
    - Sample from the posterior: f_sample = mu + sigma * z, where z ~ N(0,1)
    - Score = -f_sample / cost (higher is better: lower predicted value per unit cost)
    - This naturally balances exploration (high sigma leads to wider samples)
      and exploitation (low mu leads to lower samples), while the cost division
      favors cheaper evaluations.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Thompson sampling: draw one sample from the posterior for each candidate
    # Use a fixed seed per call? No — we want stochastic exploration.
    # But for reproducibility within a single call, use the call index or just random.
    z = np.random.standard_normal(n)
    f_sample = mu + sigma_safe * z

    # Score: prefer low predicted values and low costs
    # Higher score = more desirable
    score = -f_sample / cost_safe

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
