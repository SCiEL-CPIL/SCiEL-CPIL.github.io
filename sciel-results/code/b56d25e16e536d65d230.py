import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI)
    and Uncertainty (UCB) to balance exploitation and exploration.
    
    Parameters:
    mu: array-like, mean of the GP posterior (objective units)
    sigma: array-like, std of the GP posterior (objective units)
    f_best: float, best observed function value
    cost: array-like, cost of evaluating each candidate point
    
    Returns:
    scores: array-like, acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon to prevent division by zero and numerical instability
    eps = 1e-9
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + eps)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Uncertainty term (UCB component)
    # We use sigma directly as the uncertainty measure
    # kappa controls the exploration-exploitation trade-off
    kappa = 1.5
    ucb_term = kappa * sigma
    
    # Combined score numerator
    # Blending EI and UCB helps avoid getting stuck in local GP minima
    # and encourages exploring regions with high uncertainty
    combined_score = ei + ucb_term
    
    # Cost-aware normalization
    # We divide by cost to prioritize points that offer high improvement per unit cost
    # Add epsilon to cost to avoid division by zero (though cost >= 1 per problem statement)
    scores = combined_score / (cost + eps)
    
    # Ensure finite scores
    # If any scores are NaN or Inf (due to numerical issues), replace with a very low value
    # or 0, so they are not selected.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
