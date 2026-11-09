"""Rewritten program for cost-aware Bayesian optimization of Powell 4D."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed function value.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or very small costs (though cost >= 1 per problem description)
    # Using a small epsilon for safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # The suggested formula: (f_best - mu + sigma * 0.5) / cost
    # This balances exploitation (f_best - mu) and exploration (sigma * 0.5)
    # normalized by cost.
    scores = (f_best - mu + 0.5 * sigma) / cost_safe
    
    # Ensure all scores are finite
    # If any NaN or Inf arises (e.g., from inf mu), replace with a very low score
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
# EVOLVE-BLOCK-END
