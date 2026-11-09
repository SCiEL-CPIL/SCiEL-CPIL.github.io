"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid EI+UCB approach with dynamic exploration weight.
    For the Weierstrass function on [-0.5, 0.5], we need to balance
    exploitation (finding low values) with exploration (covering the space).
    
    The key insight: early on, we need more exploration to find the global
    minimum region. As we gather more data, we can shift toward exploitation.
    We estimate progress by comparing sigma to the range of observed values.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # Use a moderate beta that balances exploration and exploitation
    beta = 1.5
    ucb = mu - beta * sigma
    
    # Uncertainty-based score (pure exploration)
    unc = sigma
    
    # Dynamic weighting: estimate how much we've explored
    # If sigma is large relative to the spread of mu, we're still exploring
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        # Very little data or all similar values - explore more
        w_ei = 0.3
        w_ucb = 0.7
    else:
        # Normal case - balance based on relative uncertainty
        # Higher average sigma means we should explore more
        avg_sigma = np.mean(sigma)
        if avg_sigma > mu_range * 0.5:
            w_ei = 0.4
            w_ucb = 0.6
        else:
            w_ei = 0.6
            w_ucb = 0.4
    
    # Base score: weighted combination of EI and UCB
    score = w_ei * ei + w_ucb * (-ucb)
    
    # Add a small bonus for uncertainty to encourage exploration
    score += 0.15 * unc
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
