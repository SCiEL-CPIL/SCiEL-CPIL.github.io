"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive Cost-UCB.
    
    Combines the structural strength of adaptive UCB (from top inspiration) 
    with robust numerical handling. The exploration parameter kappa is 
    adapted based on the GP's uncertainty profile relative to the mean's 
    spread. The score is the optimistic improvement bound divided by the 
    evaluation cost raised to a power, favoring cheaper evaluations.
    
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
        
    # Ensure sigma is strictly positive to avoid division by zero or log issues
    # 1e-8 is small enough not to affect high sigma, but large enough for numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate statistics for adaptive kappa
    # std_mu: How much the predicted mean varies across the domain
    # mean_sigma: The average uncertainty of the GP
    std_mu = np.std(mu)
    mean_sigma = np.mean(sigma_safe)
    
    # Adaptive Kappa:
    # We want to explore more when the GP is uncertain relative to the structure it has seen.
    # ratio = mean_sigma / (std_mu + mean_sigma)
    # If std_mu is 0 (flat mean, no structure found), ratio is 1.0 -> kappa ~ 2.0 (Max exploration)
    # If mean_sigma is 0 (perfect confidence), ratio is 0.0 -> kappa ~ 1.0 (Min exploration)
    
    denom = std_mu + mean_sigma
    if denom < 1e-8:
        # GP is completely flat and certain? Unlikely. Default to moderate exploration.
        ratio = 1.0
    else:
        ratio = mean_sigma / denom
        
    # Map ratio [0, 1] to kappa [1.0, 2.0]
    # This provides a gentle increase in exploration as uncertainty dominates.
    kappa = 1.0 + ratio * 1.0
    
    # Gamma: Cost penalty exponent.
    # Costs are in [1, 2]. 
    # gamma=2.0 provides a strong preference for cheaper points without completely ignoring 
    # slightly more expensive ones if they have much higher potential improvement.
    gamma = 2.0
    
    # Compute the UCB numerator: Potential improvement
    # For minimization, we want to find low values.
    # Optimistic bound on improvement: f_best - mu + kappa * sigma
    # If this is negative, the point is likely worse than f_best even with uncertainty.
    # We keep negative values as they result in lower scores, which is correct.
    numerator = f_best - mu + kappa * sigma_safe
    
    # Compute cost penalty
    # Cost is at least 1.0 (1 + mean(x_scaled))
    cost_safe = np.maximum(cost, 1.0)
    
    # To avoid any potential overflow or division issues, we compute the discount
    # discount = 1 / (cost^gamma)
    # Using power is efficient for vectorized ops
    cost_discount = cost_safe ** gamma
    
    # Final Score
    # Divide by cost discount (equivalent to multiplying by 1/cost^gamma)
    score = numerator / cost_discount
    
    # Ensure finite values. Replace NaN/Inf with a very low value (0) to deprioritize
    # These can occur if mu or sigma contain inf/nan from upstream GP failures
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        # Just return what we have; operations are fast
        pass
        
    return score
    # EVOLVE-BLOCK-END
