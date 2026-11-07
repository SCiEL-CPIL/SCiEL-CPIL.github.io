# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability-weighted improvement per unit cost.
    
    Instead of standard Expected Improvement (EI), we use:
        P(mu < f_best) * (f_best - mu) / cost
    
    This formulation:
    1. Multiplies by P(mu < f_best) = Phi(z) where z = (f_best - mu)/sigma
       - This down-weights points where improvement is unlikely
    2. Uses (f_best - mu) as the improvement magnitude (not EI which integrates over sigma)
    3. Divides by cost for cost-awareness
    
    This is a stochastic knapsack-style objective: maximize expected regret reduction
    per unit cost, where the expectation is truncated to only count scenarios where
    improvement actually occurs.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability that mu < f_best (i.e., improvement is possible)
    prob_improve = norm.cdf(z)
    
    # Expected improvement magnitude (conditional on improvement being possible)
    # We use (f_best - mu) as a proxy for the improvement, weighted by probability
    # This is more conservative than full EI because it doesn't add the sigma*pdf(z) term
    # which represents the expected gain from the tail of the distribution
    improvement_magnitude = (f_best - mu)
    
    # Probability-weighted improvement
    prob_weighted_improvement = prob_improve * improvement_magnitude
    
    # Cost-aware: divide by cost
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = prob_weighted_improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to avoid extreme values
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
