"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement (EI) for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei


def _pi(mu, sigma, f_best):
    """Probability of Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _ucb(mu, sigma, beta):
    """Compute UCB for minimization (lower is better, so we maximize -mu + beta*sigma)."""
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses a blend of Probability of Improvement (PI) and Expected Improvement (EI),
    both divided by cost. PI provides a bounded, sigmoidal signal that naturally
    saturates in high-uncertainty regions, while EI provides a stronger exploitation
    signal. The blend is weighted to favor PI for its robustness to noise.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute PI: bounded in [0, 1], sigmoidal
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)

    # Compute EI: unbounded but typically small when improvement is small
    improvement = f_best - mu
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Normalize EI to be comparable to PI.
    # The maximum possible EI for a given sigma is roughly sigma * 0.4 (at z ~ 0.8).
    # We scale EI so that its typical magnitude is comparable to PI.
    # A simple heuristic: divide EI by (f_best - min(mu) + 1e-9) to normalize by the
    # scale of improvements, but this can be unstable. Instead, we use a fixed
    # scaling that works well for standardized GP outputs.
    # Since targets are standardized, f_best is in standardized units.
    # A reasonable scale for EI is around 0.1-0.5 for typical GP variances.
    ei_scaled = ei / (0.1 + sigma_safe.mean())

    # Blend: weight PI more heavily for robustness
    # PI is in [0,1], EI_scaled is roughly in [0, 1] for typical values
    combined = 0.7 * pi + 0.3 * ei_scaled

    # Cost-aware normalization
    scores = combined / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
