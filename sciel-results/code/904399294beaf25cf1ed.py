"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a blend of Expected Improvement (EI) and UCB, normalized by cost.
    Tuned for exploitation with some exploration.
    """
    # Ensure finite inputs; replace non-finite with safe defaults
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize: replace non-finite values
    # If mu is not finite, it's likely a numerical issue, default to f_best
    mu = np.where(np.isfinite(mu), mu, f_best)
    # If sigma is not finite or too small, default to a small positive value
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.clip(sigma, 1e-9, None)
    
    # Cost should be positive
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Handle large z to avoid overflow in pdf/cdf if necessary, though norm handles it well
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.clip(ei, 0.0, None)

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, so we want low mu and high sigma.
    # UCB_score = -mu + beta * sigma
    beta = 1.0
    ucb_score = -mu + beta * sigma

    # Blend: 80% EI, 20% UCB
    # This favors exploitation (EI) while retaining some exploration (UCB)
    blended = 0.8 * ei + 0.2 * ucb_score

    # Cost-aware: divide by cost
    # Higher score is better. Lower cost is better.
    score = blended / cost

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
