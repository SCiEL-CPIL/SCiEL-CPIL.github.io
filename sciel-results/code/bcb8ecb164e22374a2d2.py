# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a hybrid of Expected Improvement and a UCB-like term to balance
    exploitation and exploration, with a time guard.
    """
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB term: encourage exploration in areas with high uncertainty
    # Use a temperature that decays or is fixed
    # For 1D ST, the function has multiple local minima.
    # A moderate UCB weight helps escape local optima.
    ucb_weight = 0.5
    ucb = mu - ucb_weight * sigma  # Lower is better, so we minimize mu - beta*sigma
    # But we want to maximize the score. So score = -ucb = -mu + beta*sigma
    # Wait, the standard UCB for minimization is: x = argmin (mu - beta*sigma)
    # So the score should be high where (mu - beta*sigma) is low.
    # score = -(mu - beta*sigma) = -mu + beta*sigma
    # Let's combine EI and UCB.
    # EI is already a good balance. Let's try a slightly more aggressive exploration.
    
    # Alternative: Lower Confidence Bound (LCB) for minimization
    # lcb = mu - beta * sigma
    # We want to maximize -lcb = -mu + beta * sigma
    # Let's try a mix.
    
    # Let's stick to EI but maybe adjust the beta or use a different acquisition.
    # The standard EI is robust. Let's try to add a small exploration bonus.
    # score = ei + alpha * sigma
    alpha = 0.1
    score = ei + alpha * sigma

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)

    # Time check (unlikely to trigger for vectorized ops on 2000 points, but good practice)
    if time.time() > deadline:
         # Return a simple fallback if we somehow took too long (shouldn't happen)
         return -mu

    return score
# EVOLVE-BLOCK-END
