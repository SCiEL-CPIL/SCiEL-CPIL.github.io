import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Weighted Upper Confidence Bound (UCB) for minimization.
    # This approach uses a balance between exploitation (low mu) and exploration (high sigma).
    # Score = (mu - beta * sigma) / cost
    # Higher score indicates a more promising point to evaluate next.
    
    beta = 1.0
    
    # Calculate UCB values: we want to minimize mu, so we subtract beta * sigma
    # to encourage exploration in uncertain regions.
    ucb_values = mu - beta * sigma
    
    # Cost-aware scaling: divide by cost to prioritize cheaper evaluations.
    # Cost is guaranteed to be >= 1.0 based on the problem description.
    score = ucb_values / cost
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
