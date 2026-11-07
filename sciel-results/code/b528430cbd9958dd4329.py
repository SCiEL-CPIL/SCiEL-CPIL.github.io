import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 2.0
    scores = -mu + beta * sigma
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
