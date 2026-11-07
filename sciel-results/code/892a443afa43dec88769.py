import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Cost-Adjusted Expected Improvement (CA-EI) with a stronger cost penalty
    to maximize the number of evaluations within the budget, guided by the GP.
    
    Args:
        mu (np.ndarray): Mean of the GP predictions, shape (2000,).
        sigma (np.ndarray): Std dev of the GP predictions, shape (2000,).
        f_best (float): The best function value observed so far.
        cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
        
    Returns:
        np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI is non-negative by definition if we use the standard formula, 
    # but numerical errors might produce tiny negatives.
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Stronger Cost Penalty
    # The measured finding indicated that a stronger cost penalty (gamma=3 for UCB)
    # improved performance by forcing the algorithm to exploit the cheap regions
    # where the GP uncertainty is highest, allowing more total evaluations.
    # We apply this to EI.
    gamma = 3.0
    cost_penalty = cost ** gamma
    
    # Add a small exploration term to ensure we don't completely ignore
    # high-uncertainty points if EI is very small.
    # This helps in the early stages when f_best is high and EI is large,
    # but as f_best drops, EI shrinks, and we rely more on the cost/uncertainty balance.
    # A small lambda ensures sigma contributes to the score.
    lambda_explore = 0.05
    adjusted_ei = ei + lambda_explore * sigma_safe
    
    scores = adjusted_ei / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
