import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on powell 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Predicted mean of the GP at the 2000 candidate points
    sigma : array of shape (2000,)
        Predicted standard deviation of the GP at the 2000 candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled)
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = prefer to evaluate next)
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB) component
    # kappa controls exploration-exploitation tradeoff
    # Use a value that adapts based on the spread of sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Adaptive kappa: when uncertainty is high, explore more
    if sigma_std > 1e-10:
        kappa = 0.5 + 1.0 * (sigma_std / (sigma_mean + 1e-10))
    else:
        kappa = 1.0
    kappa = np.clip(kappa, 0.5, 3.0)
    
    ucb = mu + kappa * sigma_safe
    
    # Cost-aware scoring
    # We want to balance improvement potential with cost efficiency
    # Primary: EI normalized by cost (efficiency of improvement)
    ei_cost = ei / cost
    
    # Secondary: UCB normalized by cost (exploration efficiency)
    ucb_cost = ucb / cost
    
    # Combine EI and UCB with adaptive weights
    # When EI values are large relative to UCB, focus on exploitation
    # When they're comparable, balance both
    ei_scale = np.mean(ei_cost) + 1e-10
    ucb_scale = np.mean(np.abs(ucb_cost)) + 1e-10
    
    # Weight based on relative magnitudes
    ei_weight = ei_scale / (ei_scale + ucb_scale)
    ucb_weight = 1.0 - ei_weight
    
    score = ei_weight * ei_cost + ucb_weight * ucb_cost
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma_safe / cost
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget - safety margin before 330s limit
    elapsed = time.time() - start_time
    if elapsed > 300.0:
        # Return a simpler score to avoid timeout
        simple_score = ei / cost
        simple_score = np.nan_to_num(simple_score, nan=0.0, posinf=0.0, neginf=0.0)
        return simple_score
    
    return score
