"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for minimization.
    
    Args:
        mu: Array of mean predictions (shape 2000,).
        sigma: Array of standard deviation predictions (shape 2000,).
        f_best: Scalar, the best observed value so far.
        cost: Array of costs for each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,), higher is better.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Generate standard normal random variables for Thompson Sampling
    # This introduces stochasticity to break deterministic exploitation traps
    z = np.random.randn(len(mu))
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # v represents the predicted function value for each candidate
    v = mu + sigma * z
    
    # Calculate the improvement over the current best (f_best)
    # For minimization, improvement is f_best - v
    # We divide by cost to make it cost-aware
    # Higher score means better improvement per unit cost
    scores = (f_best - v) / cost
    
    # Handle any potential numerical issues (e.g., division by zero, though cost >= 1)
    # Replace non-finite values with -inf so they are never selected
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
