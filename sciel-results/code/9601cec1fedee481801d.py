"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s out of 330s wall-clock limit


def _check_deadline():
    """Return True if we are past the safety deadline."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: returns 2000 finite scores.
    Higher score = more likely to be evaluated next.
    
    Uses a hybrid of Expected Improvement (EI) with exploration bonus
    based on uncertainty, plus a small random perturbation for diversity.
    """
    global _START_TIME, _DEADLINE
    
    # Deadline guard
    if _check_deadline():
        # Return a safe fallback: points with lowest mu (most promising)
        # but ensure all finite
        scores = -np.where(np.isfinite(mu), mu, 0.0)
        return scores
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add an exploration term: reward high uncertainty
    # This helps escape local optima in early iterations
    # The weight decays as we have more confidence
    exploration_weight = 0.3
    exploration = exploration_weight * sigma
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Add a small random perturbation for diversity
    # This helps break ties and encourages exploration
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    perturbation = rng.normal(0, 0.01, size=scores.shape)
    scores = scores + perturbation
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
