"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling variant for minimization.
    
    Instead of a fixed beta UCB, we use a probabilistic approach:
    Sample from the posterior N(mu, sigma^2) and select the candidate
    with the lowest sampled value. This inherently balances exploration
    and exploitation without tuning hyperparameters.
    
    To make this work within a single acquisition call (which must return
    deterministic scores for all 2000 candidates), we use an Expected
    Improvement-like formulation that approximates Thompson sampling's
    behavior: we compute the probability that each candidate improves
    upon f_best, weighted by the expected improvement.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return exploitation-focused scores if we're running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize sigma to be in a reasonable range for the acquisition
    # Use a hybrid approach: combine exploitation (-mu) with exploration (sigma)
    # but with an adaptive beta based on the current best value
    
    # Compute a normalized uncertainty measure
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Adaptive beta: larger when we have little information (high sigma),
    # smaller when we have more information (low sigma)
    # This mimics Thompson sampling's natural exploration-exploitation balance
    if sigma_mean > 0:
        # Scale beta inversely with the relative uncertainty
        # When sigma is large relative to f_best, explore more
        beta = 1.5 * (1.0 + 0.5 * np.exp(-np.abs(f_best) / (sigma_mean + 1e-9)))
    else:
        beta = 1.0
    
    # Upper Confidence Bound for minimization
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Add a small regularization to encourage diversity in early stages
    # by slightly boosting high-uncertainty regions
    if sigma_std > 0:
        # Normalize sigma to [0, 1] range
        sigma_norm = (sigma - np.min(sigma)) / (sigma_max - np.min(sigma) + 1e-9) if sigma_max > np.min(sigma) else np.zeros_like(sigma)
        # Small bonus for uncertain regions
        scores += 0.1 * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
