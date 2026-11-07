import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses Upper Confidence Bound (UCB) with an adaptive exploration parameter.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Clip sigma to avoid division by zero or numerical issues, though UCB doesn't divide by sigma.
    # However, if sigma is 0, we rely purely on mu.
    
    # Adaptive exploration parameter beta.
    # Early in the optimization (high f_best), we want to explore more (higher beta).
    # Later (low f_best), we want to exploit (lower beta).
    # Assume f_best ranges from ~10-20 down to 0.
    # We scale f_best such that beta is 3.0 when f_best >= 5, and 1.0 when f_best <= 0.
    # beta = 1.0 + 2.0 * (f_best / 5.0)
    # Clip to [0, 1] to ensure beta is in [1.0, 3.0].
    
    # Handle case where f_best might be negative (shouldn't happen for Levy min 0, but safe)
    f_best_clipped = np.clip(f_best, 0.0, 10.0)
    beta = 1.0 + 2.0 * (f_best_clipped / 5.0)
    
    # UCB for minimization:
    # We want to select points with low predicted mean (mu) and high uncertainty (sigma).
    # Score = beta * sigma - mu
    # We divide by cost to make it cost-aware.
    
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
