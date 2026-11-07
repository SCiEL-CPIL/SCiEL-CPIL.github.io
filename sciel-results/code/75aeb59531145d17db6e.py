import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling
    # Sample from the posterior distribution
    # We need to return scores where higher is better.
    # We want to pick the point with the lowest function value.
    # So we sample y ~ N(mu, sigma^2) and return -y.
    
    # To ensure reproducibility or just standard random, we use np.random.
    # The task says "seeded instances", but the acquisition function itself
    # doesn't take a seed. Using global np.random is standard.
    
    z = np.random.randn(len(mu))
    y = mu + sigma * z
    score = -y
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
