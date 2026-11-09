"""Rewritten program: Thompson Sampling acquisition for Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: draw a single sample path from the posterior
    N(mu, sigma^2) and return negative samples (lower is better for the objective,
    but we return scores where higher means evaluate next).
    
    Since the evaluator picks the candidate with the HIGHEST score, and we want
    to minimize the objective, we return negative of the sampled function values.
    This way, regions where the posterior sample predicts a low (good) function
    value get a high (positive) score, and regions with high (bad) values get
    low (negative) scores.
    
    mu: shape (2000,) - posterior mean in objective units
    sigma: shape (2000,) - posterior std in objective units
    f_best: float - best observed value so far
    
    Returns: shape (2000,) - scores, higher = evaluate next
    """
    # Ensure sigma is strictly positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Draw a single sample from the posterior at all 2000 points
    # f ~ N(mu, sigma^2)
    sample = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # We want to minimize the objective. The evaluator picks the argmax of scores.
    # So we return -sample: lower predicted f => higher score.
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
