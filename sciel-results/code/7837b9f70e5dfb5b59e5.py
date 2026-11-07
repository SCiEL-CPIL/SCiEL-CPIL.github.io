# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Approach: Cost-Weighted Lower Confidence Bound (LCB) for minimization.
    # Instead of Expected Improvement, we use the Lower Confidence Bound of the
    # function value to estimate the best possible value in each region.
    # We maximize the expected improvement potential: f_best - LCB.
    # LCB = mu - kappa * sigma
    # Potential = f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    # We divide by cost to prefer cheaper evaluations.
    # This approach is more aggressive in exploration than EI when kappa is large.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Tuning parameter for exploration. 
    # Higher kappa encourages exploring regions with high uncertainty (high sigma).
    # For Levy 2D (multimodal), we need sufficient exploration.
    kappa = 2.5
    
    # Calculate the potential improvement using the LCB
    # If mu is much larger than f_best, this can be negative.
    # We clip to 0 to ensure we don't prioritize known-bad regions.
    potential = f_best - mu + kappa * sigma_safe
    
    # Only consider regions where the lower bound is better than current best
    # (i.e., potential > 0)
    potential = np.maximum(potential, 0.0)
    
    # Normalize by cost
    score = potential / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
