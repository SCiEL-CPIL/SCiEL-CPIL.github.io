"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    mu: (2000,) array of GP means.
    sigma: (2000,) array of GP standard deviations.
    f_best: scalar, lowest observed value.
    cost: (2000,) array of evaluation costs.
    
    Returns:
    scores: (2000,) array of acquisition scores. Higher is better.
    """
    # Ensure float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Generate standard normal samples
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from the posterior distribution
    # y = mu + sigma * z
    # Add a small epsilon to sigma to prevent zero variance issues if any, 
    # though GP sigma is usually > 0.
    y = mu + sigma * z
    
    # Calculate cost-aware score
    # We want to minimize y, so we maximize -y.
    # We divide by cost to penalize expensive evaluations.
    # Higher score = better.
    scores = -y / cost
    
    # Handle potential numerical issues
    # If cost is 0, it's undefined, but min cost is 1.
    # If y is inf, score is -inf.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
