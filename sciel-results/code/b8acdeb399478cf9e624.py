# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    
    Uses a hybrid approach: Lower Confidence Bound (LCB) with adaptive beta,
    combined with Expected Improvement (EI) for exploitation.
    
    mu: array of shape (2000,) - posterior mean in objective units
    sigma: array of shape (2000,) - posterior std in objective units
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores (higher = more promising)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (pessimistic bound, higher = more promising for minimization)
    # For minimization, LCB = mu - beta * sigma
    # We want higher scores to be more promising, so we negate: score = -mu + beta * sigma
    # But we also want to balance with EI
    
    # Adaptive beta based on uncertainty relative to best
    # When sigma is large relative to the gap to f_best, explore more
    # When sigma is small, exploit more
    gap = f_best - mu  # positive where we might improve
    # beta that balances exploration and exploitation
    # Use a fixed beta tuned for this problem
    beta = 1.8
    
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both to combine them
    # EI is already in objective units (improvement)
    # LCB score is in objective units (lower bound on potential)
    
    # Combine: use EI as primary, LCB as secondary
    # Scale EI by a factor to make it comparable
    # When f_best is close to 0, EI values are small; when far, they're larger
    
    # Alternative: use a weighted combination
    # Higher weight on exploitation when we're close to optimum
    # Higher weight on exploration when uncertain
    
    # Simple approach: use EI directly, but add a small exploration bonus
    # based on sigma
    # This is essentially EI with a small UCB component
    
    # Let's try: score = EI + alpha * sigma
    # This encourages exploration of high-uncertainty regions
    alpha = 0.5
    
    score = ei + alpha * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
