"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement with soft cost penalty.
    
    PI = Phi((f_best - mu) / sigma) naturally saturates at 1.0 for certain improvements.
    Cost penalty via exponentiation: PI^(1/cost) prioritizes cheap, high-confidence points.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Clip to (0, 1) for safe exponentiation
    pi = np.clip(pi, 1e-10, 1.0 - 1e-10)
    
    # Soft cost penalty: raise PI to power 1/cost
    # When cost is low (~1), exponent is ~1, so score ≈ PI
    # When cost is high (~2), exponent is ~0.5, so score = sqrt(PI), reducing penalty for expensive points
    # This creates a multiplicative trade-off rather than additive
    scores = np.power(pi, 1.0 / cost)
    
    # Add small uncertainty-aware exploration component
    # Points with high sigma but reasonable PI get a small boost
    exploration = sigma * 0.05 * pi
    
    scores = scores + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
