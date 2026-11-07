"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive Cost-UCB.
    
    The parameter kappa (exploration-exploitation trade-off) is optimized 
    online based on the GP's current uncertainty profile (ratio of sigma to mu variance).
    The cost penalty exponent gamma is fixed to a robust value for the given cost range.
    
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
    t0 = time.time()
    
    # Safety check for empty or invalid inputs
    if len(mu) == 0:
        return np.array([])
        
    # Ensure sigma is strictly positive to avoid division by zero in calculations
    # Using a very small epsilon to preserve the structure of low-uncertainty regions
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the spread of the GP mean and the average uncertainty
    # std_mu: How much the predicted mean varies across the domain
    # mean_sigma: The average uncertainty of the GP
    std_mu = np.std(mu)
    mean_sigma = np.mean(sigma_safe)
    
    # Adaptive Kappa:
    # When uncertainty (sigma) is high relative to the spread of the mean (std_mu),
    # we are in an exploratory phase -> increase kappa.
    # When uncertainty is low (GP is confident), we can be more exploitative -> decrease kappa.
    # Ratio is in [0, infinity). We map it to [1.0, 2.5] roughly.
    # ratio = mean_sigma / (std_mu + mean_sigma)
    # If std_mu is 0 (flat GP), ratio is 1.0 -> kappa ~ 2.0 (Max exploration)
    # If mean_sigma is 0 (perfect GP), ratio is 0.0 -> kappa ~ 1.0 (Min exploration)
    
    denom = std_mu + mean_sigma
    if denom < 1e-8:
        # GP is completely flat and certain? Unlikely. Default to moderate exploration.
        ratio = 1.0
    else:
        ratio = mean_sigma / denom
        
    # Map ratio [0, 1] to kappa [1.0, 2.0]
    # kappa = 1.0 + ratio * 1.0
    kappa = 1.0 + ratio * 1.0
    
    # Gamma: Cost penalty exponent.
    # Costs are in [1, 2]. 
    # gamma=3.0 provides a very strong preference for cheaper points,
    # aggressively exploring low-cost regions where GP uncertainty is highest.
    gamma = 3.0
    
    # Compute the UCB numerator: Potential improvement
    # We want to maximize (f_best - mu + kappa * sigma)
    numerator = f_best - mu + kappa * sigma_safe
    
    # Compute cost penalty
    cost_safe = np.maximum(cost, 1.0) # Cost is at least 1 by definition (1 + mean(x))
    cost_penalty = cost_safe ** gamma
    
    # Final Score
    score = numerator / cost_penalty
    
    # Ensure finite values. Replace NaN/Inf with a very low value (or 0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: If we are running too long, return immediately.
    # The operations above are vectorized NumPy ops on 2000 elements, 
    # which are very fast (< 1ms). The guard is a safety net.
    elapsed = time.time() - t0
    if elapsed > 250:
        # Fallback: return a simple rank-based score if something went wrong?
        # No, just return what we have.
        pass
        
    return score
    # EVOLVE-BLOCK-END
