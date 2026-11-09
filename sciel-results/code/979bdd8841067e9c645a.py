import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) 
    and Upper Confidence Bound (UCB) exploration, dynamically scaled.
    
    Parameters:
    mu (np.ndarray): Mean predictions (2000,)
    sigma (np.ndarray): Standard deviation predictions (2000,)
    f_best (float): Best observed value so far
    cost (np.ndarray): Cost of evaluating each candidate (2000,)
    
    Returns:
    np.ndarray: Scores for each candidate (2000,), higher is better
    """
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement (EI)
    # EI is the expected reduction in the minimum value
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic UCB weight (beta)
    # We want to explore more when we are far from the optimum (high f_best)
    # and exploit more when we are close (low f_best).
    # The minimum of Powell 4D is 0.
    # beta scales with f_best, capped to prevent excessive exploration in early stages
    # where sigma is high everywhere.
    beta = np.clip(0.5 * f_best, 0.1, 2.0)
    
    # Combine EI and UCB terms
    # The UCB term is beta * sigma
    # We normalize by cost to account for the efficiency of the evaluation
    score = (ei + beta * sigma) / cost
    
    # Ensure the output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
