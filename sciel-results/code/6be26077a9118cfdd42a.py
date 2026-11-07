import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of normalized EI and UCB,
    with budget-aware cost penalty.
    
    Parameters:
    mu (np.array): GP mean for 2000 candidates, shape (2000,)
    sigma (np.array): GP standard deviation for 2000 candidates, shape (2000,)
    f_best (float): Best observed function value
    cost (np.array): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.array: Scores for the 2000 candidates, higher is better.
    """
    # Track time to ensure we finish within deadline
    t_start = time.time()
    deadline = 325.0  # safety margin before 330s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate UCB (Upper Confidence Bound) for minimization
    # UCB = mu - beta * sigma (lower is better, so we want low mu and high sigma)
    # But for acquisition (higher is better), we use: mu - beta*sigma is bad,
    # so we want to maximize -mu + beta*sigma, or equivalently minimize mu - beta*sigma
    # Let's use: score_ucb = -mu + beta * sigma (higher is better for selection)
    beta = 2.5
    ucb_score = -mu + beta * sigma_safe
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute mean cost of the candidate batch
    mean_cost = np.mean(cost_safe)
    
    # Normalize EI by its scale
    ei_scale = np.std(ei) + 1e-9
    normalized_ei = ei / ei_scale
    
    # Normalize UCB score by its scale
    ucb_scale = np.std(ucb_score) + 1e-9
    normalized_ucb = ucb_score / ucb_scale
    
    # Combine EI and UCB with equal weight (or slightly favor UCB for exploration)
    combined = 0.4 * normalized_ei + 0.6 * normalized_ucb
    
    # Global budget-aware scaling: subtract penalty proportional to cost relative to mean cost
    cost_penalty = 0.3 * (cost_safe / mean_cost - 1.0)
    
    # Combine normalized score with cost penalty
    score = combined - cost_penalty
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check if we're close to the deadline
    if time.time() - t_start > deadline:
        # Return best result found so far (which is the score array)
        return score
    
    return score
