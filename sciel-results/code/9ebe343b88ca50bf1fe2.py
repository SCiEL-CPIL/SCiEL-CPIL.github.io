"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with adaptive exploration bonus.
    # Blend EI with a UCB-like exploration term to encourage diversity.
    # The exploration weight adapts based on how well we've done so far
    # (f_best close to 0 means we're near optimum, so exploit more).
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term
    # Scale exploration by uncertainty
    ucb = mu - 1.0 * sigma_safe  # Lower is better for minimization
    
    # Adaptive blend: when f_best is close to 0 (optimum), weight more on EI
    # When f_best is far from 0, weight more on exploration
    # f_best ranges from ~0 to some positive value
    # Use a smooth transition based on |f_best|
    exploration_weight = 0.5 * np.exp(-2.0 * abs(f_best))
    
    # Combine: higher score = more promising
    # EI is already "improvement" (positive when mu < f_best)
    # For UCB, we want low mu and high sigma, so score = -mu + beta*sigma
    # But we need consistent direction: higher score = evaluate next
    
    # Normalize both to comparable scales
    # EI is in objective units, UCB term also in objective units
    
    # Use a simple blend
    score = (1.0 - exploration_weight) * ei + exploration_weight * (f_best - mu + 1.5 * sigma_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
