"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Weighted Probability of Improvement (W-PoI)
    # Structural change: Instead of pure PoI (probability) or EI (expected value),
    # we use the product of the potential improvement (f_best - mu) and the
    # probability of that improvement occurring (Phi(z)).
    # This encourages moving towards points that are both likely to be better
    # and have a significant margin of improvement, balancing exploration
    # and exploitation more effectively than PoI alone for multi-modal landscapes.
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Potential Improvement (positive if mu < f_best)
    # We clip the improvement to be non-negative? No, if mu > f_best, improvement is negative.
    # W-PoI = (f_best - mu) * Phi(z)
    # If mu > f_best, (f_best - mu) is negative, Phi(z) is small (<0.5). Product is negative.
    # If mu < f_best, (f_best - mu) is positive, Phi(z) is large (>0.5). Product is positive.
    # This naturally penalizes points that are worse than current best.
    
    scores = (f_best - mu) * poi
    
    # To ensure we always have a valid score and prefer exploration if all are bad,
    # we can add a small epsilon or normalize.
    # However, the task requires "higher meaning evaluate next".
    # If all scores are negative, we pick the least negative one.
    
    # Add a small exploration bonus based on sigma to handle flat regions or
    # to ensure diversity if the W-PoI is very flat.
    # Using a smaller coefficient than before to keep W-PoI dominant.
    exploration_bonus = 0.1 * sigma
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
