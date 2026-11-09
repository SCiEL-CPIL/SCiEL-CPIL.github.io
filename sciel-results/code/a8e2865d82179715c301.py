import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB parameter. 
    # Adaptive beta based on f_best? 
    # Let's try a fixed beta first, it's more robust.
    # Beta = 2.5 is a good start for 2D problems.
    beta = 2.5
    
    # Calculate UCB for minimization: we want to maximize (beta * sigma - mu)
    # If mu is low, score is high. If sigma is high, score is high.
    ucb = beta * sigma - mu
    
    # Divide by cost
    # Ensure cost is not zero (it's >= 1)
    scores = ucb / cost
    
    # Handle any numerical issues (should be finite)
    return scores
