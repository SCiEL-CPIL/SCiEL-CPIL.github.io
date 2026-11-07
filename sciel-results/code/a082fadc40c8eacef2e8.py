# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-based acquisition function.
    
    Samples a latent function value for each candidate from the GP predictive
    distribution and selects the candidate that maximizes the negative sampled
    value divided by its evaluation cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviation predictions
        f_best: lowest observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to avoid numerical issues (division by zero or negative sigma)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Generate standard normal noise for Thompson Sampling
    # Using global np.random to ensure stochastic exploration in each step
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample latent function values
    # y_k = mu_k + sigma_k * z_k
    y = mu + sigma * z
    
    # Since we're minimizing, lower y is better.
    # We want to maximize the score, so we use -y / cost.
    # This balances finding low values (exploitation) with uncertainty (exploration via sigma)
    # and penalizes high cost.
    scores = -y / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
