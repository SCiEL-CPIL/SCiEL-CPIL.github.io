import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Blends Expected Improvement (EI), Probability of Improvement (PI),
    and a sigma-based exploration bonus, all normalized and cost-adjusted.
    """
    t_start = time.time()
    DEADLINE = 290.0  # safety margin before 300s kill

    # Convert inputs to numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Ensure positive values where needed
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Compute z-scores: how many sigma above f_best is mu (positive = likely better)
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)

    # Expected Improvement: E[max(f_best - f, 0)]
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement: P(f < f_best)
    pi = norm.cdf(z_clipped)

    # Upper Confidence Bound (UCB) for exploration: mu - kappa * sigma
    # Use a small kappa since we have the separate sigma bonus
    kappa = 0.5
    ucb = mu - kappa * sigma_safe

    # Normalize each component to [0, 1] using min-max with robust handling
    def normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        if a_max - a_min < 1e-12:
            return np.zeros_like(arr)
        return (arr - a_min) / (a_max - a_min)

    # For EI and PI, higher is better (we want high EI and high PI)
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)

    # For UCB, lower mu is better (we're minimizing), so invert
    ucb_inv = -ucb
    ucb_norm = normalize(ucb_inv)

    # Exploration bonus based on sigma (normalized)
    sigma_norm = normalize(sigma_safe)

    # Blend: EI (exploitation), PI (smooth exploitation), UCB (exploration), sigma (pure exploration)
    # Weights tuned to balance exploitation and exploration
    w_ei = 0.40
    w_pi = 0.25
    w_ucb = 0.20
    w_sigma = 0.15

    blended = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm + w_sigma * sigma_norm

    # Cost-aware: divide by cost (higher cost = lower priority)
    # Use a softened cost to avoid extreme penalties
    cost_soft = cost_safe ** 0.7
    scores = blended / cost_soft

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Time guard: return immediately if we're approaching the deadline
    if time.time() - t_start > DEADLINE - 10.0:
        return scores

    return scores
