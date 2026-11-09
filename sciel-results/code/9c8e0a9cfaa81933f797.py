"""Rewritten program using a different acquisition approach: UCB with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use an adaptive Upper Confidence Bound (UCB) style acquisition for minimization.
    We transform to a maximization problem by negating the objective.
    
    For minimization, we want to select points with low mu and high uncertainty.
    UCB for minimization: score = -mu + kappa * sigma
    where kappa is an adaptive parameter.
    
    To make it more robust, we can use a variant that balances exploitation
    (low mu) and exploration (high sigma) with an adaptive kappa based on
    the scale of mu and sigma.
    
    We also add a small expected improvement component to ensure we still
    exploit when confident, but primarily use UCB for exploration.
    
    Actually, let's use a pure UCB with a well-chosen kappa. The key insight
    is that UCB can be more effective than EI in some scenarios, especially
    when the search space is small and we have limited evaluations.
    
    For 30 total evaluations (5 initial + 25 optimization steps), we have a
    relatively small budget. UCB with a fixed or slowly decaying kappa might
    work well.
    
    Let's use kappa = 2.0, which is a common choice. We can also try to
    adapt kappa based on the ratio of sigma to the range of mu.
    """
    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB: use a kappa that balances exploration and exploitation
    # We use a fixed kappa of 2.5 for a good balance in this problem size
    kappa = 2.5
    
    # UCB score for minimization: we want to maximize -mu + kappa * sigma
    score = -mu + kappa * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
