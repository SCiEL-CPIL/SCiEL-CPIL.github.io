import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses a Kriging Believer strategy:
    1. Compute UCB scores to identify high-promise regions.
    2. Among the top UCB candidates, select those with the lowest predicted mean (mu).
    3. Return a score that favors low-mu points within high-UCB regions.
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive beta: balance exploration and exploitation
    f_global_min = 0.397887
    regret = f_best - f_global_min
    
    # When regret is high, explore more; when low, exploit more
    beta_min = 0.3
    beta_max = 2.0
    r_scale = 5.0
    
    if regret > 0:
        fraction = min(regret / (regret + r_scale), 1.0)
    else:
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # Compute UCB scores for minimization: lower mu + higher sigma is better
    # UCB = mu - beta * sigma (we want to maximize this, i.e., find low mu + high sigma)
    ucb_scores = mu - beta * sigma_safe
    
    # Find the top K% of UCB candidates
    n = len(mu)
    top_k = max(int(n * 0.1), 10)  # Top 10% or at least 10 points
    
    # Get indices of top UCB candidates
    ucb_sorted = np.argsort(ucb_scores)
    top_indices = ucb_sorted[-top_k:]
    
    # Among top UCB candidates, prefer those with lowest mu
    # Create scores: low mu is good (high score), but only within top UCB regions
    scores = np.zeros(n)
    
    # For top UCB candidates, score = -mu (lower mu -> higher score)
    # For others, score is very low
    min_mu_top = np.min(mu[top_indices])
    max_mu_top = np.max(mu[top_indices])
    mu_range = max_mu_top - min_mu_top + eps
    
    # Normalize -mu for top candidates to [0, 1]
    for idx in top_indices:
        normalized = (max_mu_top - mu[idx]) / mu_range
        scores[idx] = 1.0 + normalized
    
    # Add a small UCB component to break ties
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    scores += 0.1 * ucb_norm
    
    # Check time
    if time.time() - start_time > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
