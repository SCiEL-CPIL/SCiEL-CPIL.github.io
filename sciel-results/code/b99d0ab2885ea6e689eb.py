import numpy as np

def acquisition(mu, sigma, f_best, cost):
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clipped UCB
    # We want to minimize mu - beta*sigma.
    # Improvement potential: f_best - (mu - beta*sigma) = f_best - mu + beta*sigma
    ucb_improvement = f_best - mu + 1.0 * sigma_safe
    
    # Clip negative improvements to 0
    # If the UCB value is worse than f_best, we don't expect improvement
    score = np.maximum(0, ucb_improvement) / np.maximum(cost, 1e-9)
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
