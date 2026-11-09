"""Rewritten program: hybrid acquisition with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    with a UCB-like exploration term. Returns 2000 scores.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: encourages exploration in high-uncertainty regions
    # beta controls the exploration-exploitation tradeoff
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Combine: use a weighted sum. 
    # EI is in objective units, UCB is in objective units.
    # We normalize EI to be comparable to UCB scale if needed,
    # but for now, let's try a simple blend.
    # Since EI can be zero when mu > f_best, and UCB can be negative,
    # we want higher scores to mean "more promising".
    
    # Let's try: score = EI + 0.5 * (-mu + 0.5*sigma)
    # Actually, let's just use a tuned EI with a small exploration bonus.
    
    # Alternative: Use EI but add a small term that favors lower sigma
    # when mu is close to f_best.
    
    # Let's stick with a robust EI variant:
    # Standard EI for minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # To encourage exploration, we can scale EI by a factor that increases
    # when sigma is large, but that's already in the formula.
    
    # Let's try a different approach: use the mean of EI and a UCB variant.
    # UCB for minimization: -mu + beta * sigma
    # We want to maximize this.
    
    # Let's compute both and take a weighted average.
    # But they have different scales. Let's normalize.
    
    # Simpler: just use EI, but with a slight modification.
    # The issue with pure EI is it can be too greedy.
    
    # Let's try: score = ei + 0.1 * sigma_safe
    # This adds a small exploration bonus.
    
    score = ei + 0.1 * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
