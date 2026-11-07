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
    # Use the maximum sigma as a reference for the current uncertainty level
    max_sigma = np.max(sigma)
    
    # Beta scales with the uncertainty: when uncertainty is high, explore more
    # This ensures aggressive exploration early on and exploitation later
    beta = 2.5 * np.sqrt(np.log(1.0 + max_sigma))
    
    # Compute LCB-style optimistic gap: f_best - mu + beta * sigma
    # This is positive when we expect improvement (mu < f_best) or high uncertainty
    optimistic_gap = (f_best - mu) + beta * sigma
    
    # Cost-normalized score: prefer high optimistic gap per unit cost
    # Using sqrt(cost) might be less penalizing for expensive points if the gap is huge,
    # but standard is linear. Let's stick to linear but ensure stability.
    scores = optimistic_gap / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
