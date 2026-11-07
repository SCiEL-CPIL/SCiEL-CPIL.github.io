import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Std dev of the GP posterior, shape (2000,).
    f_best (float): Best observed function value.
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    # UCB balances exploitation (-mu) and exploration (sigma).
    # We maximize -mu + beta * sigma.
    # Dividing by cost makes the acquisition cost-aware.
    # Beta is chosen to encourage exploration in the multimodal Shekel landscape.
    beta = 1.5
    
    # Calculate UCB scores
    ucb_scores = -mu + beta * sigma
    
    # Apply cost penalty
    scores = ucb_scores / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
