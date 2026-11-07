import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -(mu - beta * sigma)
    # We want to maximize the score, which corresponds to minimizing (mu - beta * sigma)
    # A beta of 1.5 is a reasonable starting point for small budgets (25 steps)
    beta = 1.5
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute UCB values (lower is better for the objective)
    ucb_values = mu - beta * sigma
    
    # Return negative UCB values to convert minimization to maximization
    return -ucb_values
