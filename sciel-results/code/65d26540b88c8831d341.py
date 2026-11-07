"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-adjusted LCB: exploration scaled by 1/cost
    # This penalizes high-cost regions more heavily in the uncertainty bonus
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta parameter for exploration
    beta = 1.5
    
    # LCB score: we want to minimize, so lower is better
    # But acquisition should return higher = evaluate next
    # So we use negative LCB or flip the sign appropriately
    # LCB = mu - beta * sigma
    # Higher LCB means more uncertain/better candidate
    # But we want to minimize, so we actually want low mu + high uncertainty
    
    # For minimization, the acquisition function should favor low mu and high sigma
    # A common approach: score = -mu + beta * sigma (higher is better for acquisition)
    # But we need to account for cost
    
    # Cost-adjusted: scale the exploration term by 1/cost
    # High cost regions get less exploration bonus
    cost_safe = np.maximum(cost, 1e-9)
    
    # The mean term remains invariant, exploration is cost-adjusted
    score = -mu + beta * sigma / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
