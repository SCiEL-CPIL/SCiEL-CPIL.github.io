import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (2000,), GP mean predictions.
        sigma: array of shape (2000,), GP standard deviation predictions.
        f_best: float, best observed function value so far.
        cost: array of shape (2000,), cost of evaluating each candidate.
        
    Returns:
        array of shape (2000,), acquisition scores (higher is better).
    """
    # Deadline guard (safety margin)
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + eps)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    # We add a term proportional to sigma to encourage exploration.
    # Beta controls the trade-off. 1.5 is a reasonable default for multimodal problems.
    beta = 1.5
    exploration = beta * sigma
    
    # Combine EI and Exploration
    # The hint suggests "Adding exploration bonus to cost-normalized EI"
    # This can be interpreted as (EI + Exploration) / Cost
    # Or EI / Cost + Exploration / Cost
    # They are mathematically identical.
    
    # Calculate Cost-Weighted Score
    # We divide by cost to prefer cheaper points if scores are similar.
    # We add eps to cost to avoid division by zero (though cost >= 1).
    score = (ei + exploration) / (cost + eps)
    
    # Check deadline
    if time.time() > deadline:
        # If we are running out of time, return the current scores.
        # The calculation above is vectorized and fast, so this is unlikely to be needed.
        pass
        
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

# EVOLVE-BLOCK-END
