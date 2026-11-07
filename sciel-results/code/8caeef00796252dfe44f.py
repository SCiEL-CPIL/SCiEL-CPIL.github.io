import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB acquisition function
    # beta controls exploration vs exploitation
    beta = 2.5
    
    # Ensure sigma is not zero to avoid issues, though UCB doesn't divide by it
    # But we use it in the term
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB score: -mu + beta * sigma
    # We want to maximize this score (find low mu, high sigma)
    # Penalize cost by subtracting it
    score = (-mu + beta * sigma_safe) - cost
    
    return score
