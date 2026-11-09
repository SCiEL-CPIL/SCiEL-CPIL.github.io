"""Thompson Sampling proxy acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling proxy acquisition.
    
    Instead of selecting a single acquisition function from a portfolio based on
    heuristic metrics, we compute a score that approximates the expected value
    of a sample drawn from the posterior, normalized by the uncertainty.
    
    For minimization:
    1. We want low mu (exploitation).
    2. We want high sigma (exploration), but scaled appropriately.
    
    The score is defined as:
    score = (f_best - mu) / (sigma + eps)
    
    This is effectively a normalized Expected Improvement (without the PDF term 
    for simplicity and stability, or as a proxy for the z-score in EI).
    Higher scores indicate points that are significantly better than f_best 
    relative to their uncertainty.
    
    This structural change from linear combinations (like UCB/LCB) to a 
    normalized ratio encourages exploration in uncertain regions while 
    strongly exploiting regions where the mean is already very low.
    """
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = mu.shape[0]
    
    # Guard against division by zero / near-zero sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the normalized improvement score
    # This acts as a z-score for the potential improvement
    # If mu < f_best, the numerator is positive.
    # Dividing by sigma normalizes by uncertainty.
    # This naturally balances:
    # - Low mu (large positive numerator)
    # - High sigma (denominator reduces score, but if mu is much lower than f_best, 
    #   the ratio remains high. If mu is close to f_best, high sigma reduces score, 
    #   preventing pure random exploration when no improvement is likely.)
    
    # A common Thompson Sampling approach is to sample theta ~ N(mu, sigma) 
    # and pick min(theta). 
    # The expected value of min(theta) over N samples is roughly 
    # mu - sigma * E[min(Z_i)] where Z_i ~ N(0,1).
    # For N=2000, E[min(Z_i)] is approx -3.0 to -3.5.
    # So minimizing theta is equivalent to maximizing -mu + c*sigma.
    # However, the "normalized improvement" (f_best - mu)/sigma is a standard 
    # proxy that has performed well in practice for balancing the trade-off 
    # without needing to tune a beta parameter.
    
    # Let's use a slightly more robust version:
    # score = (f_best - mu) / sigma_safe
    
    # To ensure we don't ignore points with very low mu but moderate sigma, 
    # and to handle the case where f_best is very low (making f_best - mu 
    # negative for most points), we can add a small constant or use 
    # a different normalization.
    
    # Actually, if f_best is the global minimum found so far, 
    # f_best - mu is positive only where we expect improvement.
    # If mu > f_best everywhere, all scores are negative.
    # We should still pick the "least bad" or most promising one.
    
    # Let's stick to the normalized improvement.
    scores = (f_best - mu) / sigma_safe
    
    # Sanitize to ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check time
    if time.time() > DEADLINE - 5:
        return _sanitize(scores)
    
    return _sanitize(scores)


def _sanitize(scores):
    """Ensure all scores are finite."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    return scores
# EVOLVE-BLOCK-END
