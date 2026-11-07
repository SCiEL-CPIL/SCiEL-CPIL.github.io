import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto-filtered UCB.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score for minimization: mu - beta * sigma
    # Higher score is better (lower value)
    # We want to minimize, so we want low mu and high sigma (uncertainty)
    # Score = mu - beta * sigma. We maximize this score?
    # No, if we minimize f, we want f to be small.
    # If we use UCB = mu - beta*sigma, a low mu and high sigma gives a very negative score.
    # If we maximize the score, we pick the most negative one?
    # Usually for minimization, UCB is defined as mu - beta*sigma and we MINIMIZE it.
    # But the contract says "higher meaning evaluate next".
    # So we should return scores where higher is better.
    # If we minimize f, we want candidates with low predicted f.
    # So score should be high when mu is low and sigma is high.
    # Score = -mu + beta * sigma ?
    # Let's check EI. EI is positive when mu < f_best.
    # If mu is very low, EI is high.
    # So for UCB, we want a score that is high for low mu and high sigma.
    # Score = -mu + beta * sigma.
    
    # Let's re-read the directive: "applies a pure, unnormalized Upper Confidence Bound (UCB) score"
    # Usually UCB for minimization is mu - beta*sigma, and we select the min.
    # To convert to "higher is better", we can negate it: -mu + beta*sigma.
    
    beta = 2.0
    ucb_scores = -mu + beta * sigma
    
    # Filtering strategy
    # Mask out candidates where mu is worse than f_best by more than tolerance
    # "Worse" means mu > f_best
    # Tolerance: small
    tol = 0.05
    # If f_best is very small, tol should be small.
    # If f_best is large, tol can be larger?
    # Let's use a relative tolerance as well.
    tol = max(0.05, 0.1 * f_best)
    
    mask = mu <= f_best + tol
    
    # If all are masked, keep the one with the lowest mu
    if not np.any(mask):
        min_idx = np.argmin(mu)
        mask[min_idx] = True
    
    # Apply mask
    scores = np.full_like(mu, -1e10, dtype=np.float64)
    scores[mask] = ucb_scores[mask]
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
