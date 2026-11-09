import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: minimize mu - beta * sigma
    # Score is higher for better candidates.
    # So score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Adaptive beta? 
    # If we assume early stages have higher sigma, a static beta works well.
    # Let's try beta = 2.0 to encourage exploration of the 5 local minima.
    
    beta = 2.0
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
