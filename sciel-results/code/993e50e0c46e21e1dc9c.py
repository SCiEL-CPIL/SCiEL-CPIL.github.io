import numpy as np

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    eps = 1e-9
    sigma = np.maximum(sigma, eps)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to find points with low function values.
    # The Lower Confidence Bound (LCB) is mu - beta * sigma.
    # We want to minimize LCB, so we maximize -LCB = beta * sigma - mu.
    
    # Beta controls the exploration/exploitation trade-off.
    # A value of 2.0 is a common heuristic for a small number of evaluations.
    beta = 2.0
    
    scores = beta * sigma - mu
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
