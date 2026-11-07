# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition for Rastrigin 2D minimization.
    
    Maximizes (f_best - mu + beta * sigma) / cost, where beta is dynamically
    scaled to balance exploration of high-variance, low-cost regions.
    
    Parameters:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP standard deviation predictions in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues and negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative values
    cost = np.maximum(cost, 1e-6)
    
    # Dynamic beta: scale exploration based on current uncertainty
    # Use the median sigma as a reference for the current uncertainty level
    med_sigma = np.median(sigma)
    
    # Beta scales with the uncertainty: when uncertainty is high, explore more
    # The factor 2.0 is a reasonable starting point for Rastrigin's multimodality
    # We also add a small constant to ensure some exploration even when sigma is small
    beta = 2.0 + 1.0 * np.sqrt(np.log(1.0 + med_sigma / (1e-6 + np.max(sigma))))
    
    # Compute LCB-style optimistic gap: f_best - mu + beta * sigma
    # This is positive when we expect improvement (mu < f_best) or high uncertainty
    optimistic_gap = (f_best - mu) + beta * sigma
    
    # Cost-normalized score: prefer high optimistic gap per unit cost
    scores = optimistic_gap / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
