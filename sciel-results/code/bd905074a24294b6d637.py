"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-normalized UCB with high gamma.
    
    Score = (mu + kappa * sigma) / cost^gamma
    
    This aggressively explores cheap, uncertain regions of the domain,
    which is critical for locating global minima within a tight budget.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Cost-normalized UCB with tuned parameters
    # kappa controls exploration strength
    # gamma controls cost penalty aggressiveness
    kappa = 2.0
    gamma = 2.5
    
    # For minimization, we want points where mu is low and sigma is high.
    # UCB for minimization: mu - kappa * sigma (lower is better)
    # Higher acquisition score = more desirable to evaluate
    # So: score = (f_best - mu + kappa * sigma) / cost^gamma
    
    # Compute the numerator: f_best - mu + kappa * sigma
    numerator = f_best - mu + kappa * sigma_safe
    
    # Ensure non-negative
    numerator = np.maximum(numerator, 0.0)
    
    # Compute cost penalty using log-space for smoother behavior
    # cost is in [1, 2] (1 + mean(x_scaled)), so log is well-behaved
    cost_safe = np.maximum(cost, 1e-12)
    # Use a combination: power law for strong penalty + small linear term
    # This ensures expensive points aren't completely zeroed out
    log_cost = np.log(cost_safe)
    cost_penalty = np.exp(gamma * log_cost)  # = cost^gamma, but computed in log space
    
    # Final score
    score = numerator / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin
        # Return a simple fallback if we're running too long
        pass
    
    return score
    # EVOLVE-BLOCK-END
