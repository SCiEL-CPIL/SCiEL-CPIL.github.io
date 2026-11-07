import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Predicted mean (shape 2000).
    sigma (np.ndarray): Predicted standard deviation (shape 2000).
    f_best (float): Best observed value.
    cost (np.ndarray): Cost of evaluating each candidate (shape 2000).
    
    Returns:
    np.ndarray: Scores (shape 2000), higher is better.
    """
    # UCB parameter for exploration
    # With few initial points, a higher beta encourages exploration of uncertain regions.
    beta = 2.0
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: We want to find low values.
    # The "optimistic" prediction for a minimizer is mu - beta * sigma.
    # We want to maximize the acquisition score, so we negate the optimistic prediction
    # or equivalently maximize (-mu + beta * sigma).
    ucb_score = -mu + beta * sigma
    
    # Cost-aware normalization
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
