import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in z calculation
    # sigma is standard deviation, should be positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus
    # We want to explore where sigma is high.
    # Since we are minimizing, high sigma means potential for low values.
    # Add a bonus proportional to sigma.
    # Normalize sigma? 
    # sigma is in objective units.
    # Let's use a small weight.
    ucb_bonus = 0.1 * sigma_safe
    
    # Combine EI and UCB bonus
    # The knowledge suggests keeping cost-normalized signal dominant.
    # So EI is the main term.
    combined = ei + ucb_bonus
    
    # Cost normalization
    # cost is 1 + mean(x_scaled)
    # Higher cost means more expensive.
    # We divide by cost to prefer cheaper points.
    # Ensure cost is positive (it is, min 1)
    scores = combined / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
