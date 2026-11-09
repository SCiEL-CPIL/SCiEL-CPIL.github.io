import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu (np.ndarray): Posterior mean of shape (2000,)
    sigma (np.ndarray): Posterior standard deviation of shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores of shape (2000,), higher is better.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # Beta controls the exploration/exploitation balance.
    # A higher beta favors exploration (high sigma).
    beta = 2.5 
    ucb = mu + beta * sigma_safe
    
    # 3. Hybrid Score
    # We combine them to leverage the local improvement signal of EI
    # and the global exploration signal of UCB.
    # Normalizing EI can help, but since they are in objective units,
    # a direct sum is often effective. We weight EI slightly higher 
    # to ensure we still converge, but UCB is strong enough to escape local optima.
    score = ei + 0.5 * ucb
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
